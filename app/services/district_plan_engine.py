import json
from typing import Dict, List, Optional
from collections import Counter
from sqlalchemy.orm import Session
from rapidfuzz import fuzz
from app.models.models import District, JobPosting, Course, EmployerFeedback, NSQFQualificationPack

FUZZY_THRESHOLD = 80

# Batch planning constants
DEFAULT_BATCH_SIZE = 30          # students per batch
DEFAULT_COURSE_DURATION_MONTHS = 6  # months per batch
LOW_ENROLMENT_THRESHOLD = 30
LOW_PLACEMENT_THRESHOLD = 40.0
HIGH_DEMAND_THRESHOLD = 5        # 5+ postings = significant demand
URGENT_GAP_THRESHOLD = 50.0     # gap% above this = urgent
MODERATE_GAP_THRESHOLD = 25.0    # gap% above this = moderate


class DistrictPlanEngine:
    """
    Generates a comprehensive, structured district-level training plan that:
      - Analyses live labour-market demand by skill, sector, and company
      - Compares against local training supply (courses + their syllabi)
      - Identifies gaps, surpluses, and courses at risk
      - Computes batch recommendations with capacity estimates
      - Incorporates employer feedback and NSQF alignment
      - Produces prioritised, time-bound action items
    """

    @staticmethod
    def _is_taught(skill: str, taught_skills: List[str]) -> bool:
        """Fuzzy-match a demanded skill against the union of all course syllabus skills."""
        skill_l = skill.lower()
        for t in taught_skills:
            if fuzz.token_set_ratio(skill_l, t.lower()) >= FUZZY_THRESHOLD:
                return True
        return False

    @staticmethod
    def _classify_demand_level(demand_count: int, total_postings: int) -> str:
        """Classify how hot a skill is relative to the district's total postings."""
        if total_postings == 0:
            return "No Data"
        pct = (demand_count / total_postings) * 100
        if pct >= 40:
            return "Critical"
        elif pct >= 20:
            return "High"
        elif pct >= 10:
            return "Moderate"
        else:
            return "Low"

    @staticmethod
    def _find_best_nsqf_qp(db: Session, skill: str) -> Optional[Dict]:
        """Find the best-matching NSQF Qualification Pack for a given skill."""
        qps = db.query(NSQFQualificationPack).all()
        best_qp = None
        best_score = 0
        for qp in qps:
            qp_skills = json.loads(qp.covered_skills_json) if qp.covered_skills_json else []
            for qp_skill in qp_skills:
                score = fuzz.token_set_ratio(skill.lower(), qp_skill.lower())
                if score > best_score:
                    best_score = score
                    best_qp = {
                        "qp_code": qp.qp_code,
                        "qp_title": qp.title,
                        "sector": qp.sector,
                        "nsqf_level": qp.nsqf_level,
                        "match_score": round(best_score, 1),
                    }
        return best_qp if best_score >= FUZZY_THRESHOLD else None

    @staticmethod
    def _get_employer_concerns(db: Session, district_name: str) -> List[Dict]:
        """Pull employer feedback for courses in this district."""
        feedbacks = db.query(EmployerFeedback).filter(
            EmployerFeedback.district_name == district_name
        ).all()
        concerns = []
        for f in feedbacks:
            missing = json.loads(f.missing_skills_json) if f.missing_skills_json else []
            validated = json.loads(f.validated_skills_json) if f.validated_skills_json else []
            concerns.append({
                "employer": f.employer_name,
                "company": f.company,
                "sector": f.sector,
                "satisfaction_rating": f.satisfaction_rating,
                "validated_skills": validated,
                "missing_skills": missing,
                "comments": f.comments,
            })
        return concerns

    @classmethod
    def generate_district_plan(cls, db: Session, district_name: str) -> Dict:
        district = db.query(District).filter(District.name == district_name).first()
        if not district:
            raise ValueError(f"District '{district_name}' not found")

        # ─── 1. DEMAND ANALYSIS ───────────────────────────────────────────
        postings = db.query(JobPosting).filter(
            JobPosting.district_name == district_name
        ).all()

        # Skill frequency
        skill_counts: Counter = Counter()
        sector_demand: Counter = Counter()
        company_demand: Counter = Counter()
        language_breakdown: Counter = Counter()

        for jp in postings:
            skills = json.loads(jp.extracted_skills_json) if jp.extracted_skills_json else []
            for s in skills:
                skill_counts[s] += 1
            sector_demand[jp.sector] += 1
            company_demand[jp.company] += 1
            language_breakdown[jp.language] += 1

        total_postings = len(postings)

        # ─── 2. SUPPLY ANALYSIS ───────────────────────────────────────────
        courses = db.query(Course).filter(
            Course.district_name == district_name
        ).all()

        taught_skills: List[str] = []
        course_summary = []
        for c in courses:
            syllabus = json.loads(c.syllabus_skills_json) if c.syllabus_skills_json else []
            taught_skills.extend(syllabus)
            course_summary.append({
                "course_id": c.id,
                "course_code": c.course_code,
                "title": c.title,
                "sector": c.sector,
                "institution_type": c.institution_type,
                "enrolment_count": c.enrolment_count,
                "placement_rate": c.placement_rate,
                "syllabus_skills": syllabus,
                "health": cls._assess_course_health(c),
            })

        # ─── 3. GAP ANALYSIS ──────────────────────────────────────────────
        skills_available = []
        skills_gap = []
        skills_declining = []  # taught but not demanded

        for skill, freq in skill_counts.most_common():
            demand_level = cls._classify_demand_level(freq, total_postings)
            entry = {
                "skill": skill,
                "demand_count": freq,
                "demand_percentage": round((freq / max(1, total_postings)) * 100, 1),
                "demand_level": demand_level,
            }
            if cls._is_taught(skill, taught_skills):
                entry["status"] = "Available"
                skills_available.append(entry)
            else:
                entry["status"] = "Gap"
                entry["nsqf_alignment"] = cls._find_best_nsqf_qp(db, skill)
                skills_gap.append(entry)

        # Skills taught but NOT demanded → oversupplied / declining
        demanded_set = {s.lower() for s in skill_counts.keys()}
        all_taught_set = set()
        for c in courses:
            syllabus = json.loads(c.syllabus_skills_json) if c.syllabus_skills_json else []
            for s in syllabus:
                all_taught_set.add(s)

        for skill in all_taught_set:
            if not cls._is_taught(skill, [s for s in skill_counts.keys()]):
                skills_declining.append({
                    "skill": skill,
                    "demand_count": 0,
                    "status": "Oversupplied — Not Demanded",
                })

        # ─── 4. COURSES AT RISK ───────────────────────────────────────────
        courses_at_risk = []
        for c in courses:
            if c.enrolment_count < LOW_ENROLMENT_THRESHOLD and c.placement_rate < LOW_PLACEMENT_THRESHOLD:
                courses_at_risk.append({
                    "course_id": c.id,
                    "course_code": c.course_code,
                    "title": c.title,
                    "sector": c.sector,
                    "enrolment_count": c.enrolment_count,
                    "placement_rate": c.placement_rate,
                    "status": "Consider Discontinuing",
                    "reason": (
                        f"Enrolment {c.enrolment_count} (below {LOW_ENROLMENT_THRESHOLD} threshold) "
                        f"and placement {c.placement_rate}% (below {LOW_PLACEMENT_THRESHOLD}% threshold)"
                    ),
                    "recommended_action": cls._recommend_course_action(c, skill_counts),
                })

        # ─── 5. BATCH PLANNING ────────────────────────────────────────────
        batch_recommendations = []
        for g in skills_gap:
            if g["demand_count"] >= HIGH_DEMAND_THRESHOLD:
                estimated_candidates = g["demand_count"] * 4  # assume 4x demand = candidates needed
                batches_needed = max(1, (estimated_candidates + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE)
                batch_recommendations.append({
                    "skill": g["skill"],
                    "demand_count": g["demand_count"],
                    "demand_level": g["demand_level"],
                    "estimated_candidates_needed": estimated_candidates,
                    "recommended_batch_size": DEFAULT_BATCH_SIZE,
                    "batches_needed": batches_needed,
                    "estimated_duration_months": DEFAULT_COURSE_DURATION_MONTHS,
                    "nsqf_alignment": g.get("nsqf_alignment"),
                    "priority": "High" if g["demand_level"] in ("Critical", "High") else "Medium",
                })

        # ─── 6. CURRICULUM UPDATE RECOMMENDATIONS ─────────────────────────
        curriculum_updates = []
        if skills_gap and courses:
            # Find the strongest course to augment
            strongest = max(courses, key=lambda c: c.placement_rate)
            top_gaps = [g["skill"] for g in skills_gap[:3]]
            curriculum_updates.append({
                "target_course": strongest.title,
                "course_code": strongest.course_code,
                "action": "Add Modules",
                "skills_to_add": top_gaps,
                "reason": f"Course has highest placement rate ({strongest.placement_rate}%) — augmenting it closes demand gaps fastest.",
                "nsqf_alignment": cls._find_best_nsqf_qp(db, top_gaps[0]) if top_gaps else None,
            })

        # Recommend deprecating obsolete skills from courses
        if skills_declining:
            for c in courses:
                syllabus = json.loads(c.syllabus_skills_json) if c.syllabus_skills_json else []
                obsolete_in_course = [
                    s for s in syllabus
                    if not cls._is_taught(s, [sk for sk in skill_counts.keys()])
                ]
                if obsolete_in_course:
                    curriculum_updates.append({
                        "target_course": c.title,
                        "course_code": c.course_code,
                        "action": "Deprecate Modules",
                        "skills_to_remove": obsolete_in_course,
                        "reason": "These skills have zero demand in current district job postings.",
                    })

        # ─── 7. EMPLOYER INTELLIGENCE ────────────────────────────────────
        employer_concerns = cls._get_employer_concerns(db, district_name)
        employer_missing_skills: Counter = Counter()
        for ec in employer_concerns:
            for ms in ec["missing_skills"]:
                employer_missing_skills[ms] += 1

        # ─── 8. DISTRICT SUMMARY SCORES ──────────────────────────────────
        total_demanded = len(skill_counts)
        total_available = len(skills_available)
        gap_ratio = (len(skills_gap) / max(1, total_demanded)) * 100
        supply_demand_ratio = round((total_available / max(1, total_demanded)) * 100, 1)

        if gap_ratio >= URGENT_GAP_THRESHOLD:
            urgency = "Critical"
        elif gap_ratio >= MODERATE_GAP_THRESHOLD:
            urgency = "High"
        else:
            urgency = "Moderate"

        # ─── 9. PRIORITISED ACTION ITEMS ─────────────────────────────────
        action_items = []

        # Priority 1: Launch batches for critical gap skills
        for br in batch_recommendations:
            if br["priority"] == "High":
                action_items.append({
                    "priority": 1,
                    "category": "New Training Batch",
                    "action": f"Launch {br['batches_needed']} batch(es) of {br['skill']} training ({br['recommended_batch_size']} students/batch, {br['estimated_duration_months']} months)",
                    "timeline": "0-3 months",
                    "expected_impact": f"Addresses {br['demand_count']} active job postings ({br['demand_level']} demand)",
                    "nsqf_qp": br.get("nsqf_alignment", {}).get("qp_code") if br.get("nsqf_alignment") else None,
                })

        # Priority 2: Discontinue / restructure at-risk courses
        for c in courses_at_risk:
            action_items.append({
                "priority": 2,
                "category": "Course Discontinuation",
                "action": f"Discontinue or restructure '{c['title']}' — redirect students to high-demand courses",
                "timeline": "1-2 months",
                "expected_impact": f"Frees up {c['enrolment_count']} seats and instructor capacity for redeployment",
            })

        # Priority 3: Update curriculum for existing courses
        for cu in curriculum_updates:
            if cu["action"] == "Add Modules":
                action_items.append({
                    "priority": 3,
                    "category": "Curriculum Update",
                    "action": f"Add {', '.join(cu['skills_to_add'][:3])} modules to '{cu['target_course']}'",
                    "timeline": "2-4 months",
                    "expected_impact": "Closes demand gap without launching entirely new courses",
                })
            elif cu["action"] == "Deprecate Modules":
                action_items.append({
                    "priority": 4,
                    "category": "Curriculum Cleanup",
                    "action": f"Remove obsolete modules from '{cu['target_course']}': {', '.join(cu['skills_to_remove'][:3])}",
                    "timeline": "Next academic cycle",
                    "expected_impact": "Reclaims instructional hours for relevant content",
                })

        # Priority 5: Employer-confirmed missing skills
        for skill, count in employer_missing_skills.most_common(3):
            action_items.append({
                "priority": 5,
                "category": "Employer-Flagged Gap",
                "action": f"Address employer-reported missing skill: '{skill}' (flagged by {count} employer(s))",
                "timeline": "0-3 months",
                "expected_impact": "Improves employer satisfaction and placement outcomes",
            })

        if not action_items:
            action_items.append({
                "priority": 5,
                "category": "Monitoring",
                "action": "Continue monitoring — district training supply is well aligned with current demand.",
                "timeline": "Ongoing",
                "expected_impact": "Maintain alignment",
            })

        # ─── 10. FINAL STRUCTURED PLAN ───────────────────────────────────
        return {
            # District context
            "district": district.name,
            "region": district.region,
            "major_industries": json.loads(district.major_industries) if district.major_industries else [],

            # Demand snapshot
            "demand_summary": {
                "total_active_postings": district.active_postings_count,
                "postings_analysed": total_postings,
                "unique_skills_demanded": total_demanded,
                "top_sectors_by_demand": [
                    {"sector": s, "posting_count": c}
                    for s, c in sector_demand.most_common(5)
                ],
                "top_companies_hiring": [
                    {"company": c, "posting_count": n}
                    for c, n in company_demand.most_common(5)
                ],
                "language_breakdown": dict(language_breakdown),
            },

            # Supply snapshot
            "supply_summary": {
                "total_courses_offered": len(courses),
                "total_enrolment_capacity": sum(c.enrolment_count for c in courses),
                "average_placement_rate": round(
                    sum(c.placement_rate for c in courses) / max(1, len(courses)), 1
                ),
                "courses_at_risk_count": len(courses_at_risk),
            },

            # Gap analysis
            "gap_analysis": {
                "skills_available": skills_available,
                "skills_gap": skills_gap,
                "skills_oversupplied": skills_declining,
                "gap_ratio_percentage": round(gap_ratio, 1),
                "supply_demand_ratio": supply_demand_ratio,
                "urgency_level": urgency,
            },

            # Courses at risk
            "courses_at_risk": courses_at_risk,
            "course_health_overview": course_summary,

            # Batch plan
            "batch_plan": batch_recommendations,

            # Curriculum actions
            "curriculum_updates": curriculum_updates,

            # Employer intelligence
            "employer_intelligence": {
                "feedback_count": len(employer_concerns),
                "top_employer_flagged_gaps": [
                    {"skill": s, "flagged_by_count": c}
                    for s, c in employer_missing_skills.most_common(5)
                ],
                "feedback_details": employer_concerns,
            },

            # Action plan
            "action_plan": {
                "total_actions": len(action_items),
                "urgency_level": urgency,
                "items": sorted(action_items, key=lambda x: x["priority"]),
            },

            # Metadata
            "plan_metadata": {
                "generated_for": "SIH 26134 — Government of Maharashtra",
                "methodology": "NLP-extracted skills from job postings, fuzzy-matched against course syllabi, weighted by demand frequency, employer feedback, and placement outcomes.",
            },
        }

    @staticmethod
    def _assess_course_health(course: Course) -> Dict:
        """Quick health assessment for a single course."""
        health_score = 0
        issues = []

        if course.enrolment_count < LOW_ENROLMENT_THRESHOLD:
            issues.append("Low enrolment")
        else:
            health_score += 30

        if course.placement_rate < LOW_PLACEMENT_THRESHOLD:
            issues.append("Poor placement")
        else:
            health_score += 40

        if course.placement_rate >= 70:
            health_score += 30

        if health_score >= 80:
            status = "Healthy"
        elif health_score >= 50:
            status = "Moderate"
        else:
            status = "At Risk"

        return {
            "health_score": health_score,
            "status": status,
            "issues": issues if issues else ["None"],
        }

    @staticmethod
    def _recommend_course_action(course: Course, skill_counts: Counter) -> str:
        """Generate a specific recommendation for an at-risk course."""
        syllabus = json.loads(course.syllabus_skills_json) if course.syllabus_skills_json else []

        # Check if any syllabus skill is in demand
        in_demand = [s for s in syllabus if skill_counts.get(s, 0) > 0]

        if not in_demand:
            return "Discontinue entirely — no syllabus skills are in demand in this district."
        elif course.placement_rate < 30:
            return "Restructure urgently — retain only in-demand modules and add emerging skills."
        else:
            return "Review and modernise — some skills still relevant but curriculum needs updating."
