import json
from typing import Dict, List, Optional
from collections import Counter
from sqlalchemy.orm import Session
from rapidfuzz import fuzz
from app.models.models import (
    District, JobPosting, Course, EmployerFeedback, NSQFQualificationPack
)

# ── Constants ────────────────────────────────────────────────────────────────
FUZZY_THRESHOLD = 80
DEFAULT_BATCH_SIZE = 30
DEFAULT_COURSE_DURATION_MONTHS = 6
LOW_ENROLMENT_THRESHOLD = 30
LOW_PLACEMENT_THRESHOLD = 40.0
HIGH_DEMAND_THRESHOLD = 5
URGENT_GAP_THRESHOLD = 50.0
MODERATE_GAP_THRESHOLD = 25.0

# Budget estimates (rough Indian context, per batch / per student)
COST_PER_STUDENT_TRAINING = 8000          # ₹ — course delivery cost per student
COST_PER_LAPTOP = 35000                   # ₹ — if IT/equipment-heavy course
COST_INSTRUCTOR_PER_MONTH = 45000         # ₹ — instructor salary per month
COURSE_DURATION_MONTHS = 6
COST_EQUIPMENT_MANUFACTURING = 200000     # ₹ — workshop equipment per batch
COST_EQUIPMENT_IT = 500000               # ₹ — computer lab setup per batch
COST_EQUIPMENT_ELECTRONICS = 150000      # ₹ — electronics lab per batch

# Sector → equipment cost mapping
SECTOR_EQUIPMENT_COST = {
    "IT-ITeS": COST_EQUIPMENT_IT,
    "Automotive": COST_EQUIPMENT_MANUFACTURING,
    "Capital Goods": COST_EQUIPMENT_MANUFACTURING,
    "Electronics": COST_EQUIPMENT_ELECTRONICS,
    "Agriculture": 80000,
    "Healthcare": 120000,
    "Logistics": 60000,
    "Retail": 40000,
    "BFSI": 30000,
    "Apparel": 70000,
}

# Emerging tech skills (PM-SETU 32 new-age trades aligned)
EMERGING_TECH_SKILLS = [
    {"skill": "AI Programming Assistant", "sector": "IT-ITeS", "horizon": "6-12 months", "pm_setu_trade": True},
    {"skill": "Drone Pilot & Maintenance", "sector": "Aerospace", "horizon": "6-12 months", "pm_setu_trade": True},
    {"skill": "Mechanic Electric Vehicle", "sector": "Automotive", "horizon": "0-6 months", "pm_setu_trade": True},
    {"skill": "Semiconductor Technician", "sector": "Electronics", "horizon": "12-18 months", "pm_setu_trade": True},
    {"skill": "5G Telecom Technician", "sector": "Telecom", "horizon": "6-12 months", "pm_setu_trade": True},
    {"skill": "Green Hydrogen Technician", "sector": "Green Jobs", "horizon": "12-18 months", "pm_setu_trade": True},
    {"skill": "Solar Panel Installation", "sector": "Green Jobs", "horizon": "0-6 months", "pm_setu_trade": False},
    {"skill": "IoT & Sensor Networks", "sector": "Electronics", "horizon": "6-12 months", "pm_setu_trade": False},
    {"skill": "3D Printing & Additive Manufacturing", "sector": "Capital Goods", "horizon": "6-12 months", "pm_setu_trade": False},
    {"skill": "Robotic Process Automation", "sector": "IT-ITeS", "horizon": "6-12 months", "pm_setu_trade": False},
    {"skill": "Data Analytics & Visualization", "sector": "IT-ITeS", "horizon": "0-6 months", "pm_setu_trade": False},
    {"skill": "Cybersecurity Fundamentals", "sector": "IT-ITeS", "horizon": "0-6 months", "pm_setu_trade": False},
]


class DistrictPlanEngine:
    """
    Generates a comprehensive district-level training plan with:
      - Live demand analysis (skill, sector, company, language)
      - Supply analysis (courses, capacity, health)
      - Gap analysis (available, gap, oversupplied)
      - Before/after projection
      - Time-bound action plan (immediate / short / medium term)
      - Budget estimates per recommendation
      - Cross-district comparison
      - Emerging technology horizon scan
      - Capacity utilisation analysis
      - Scenario planning
    """

    # ════════════════════════════════════════════════════════════════════════
    #  HELPER METHODS
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_taught(skill: str, taught_skills: List[str]) -> bool:
        skill_l = skill.lower()
        for t in taught_skills:
            if fuzz.token_set_ratio(skill_l, t.lower()) >= FUZZY_THRESHOLD:
                return True
        return False

    @staticmethod
    def _classify_demand_level(demand_count: int, total_postings: int) -> str:
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

    @staticmethod
    def _assess_course_health(course: Course) -> Dict:
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
        return {"health_score": health_score, "status": status, "issues": issues if issues else ["None"]}

    @staticmethod
    def _recommend_course_action(course: Course, skill_counts: Counter) -> str:
        syllabus = json.loads(course.syllabus_skills_json) if course.syllabus_skills_json else []
        in_demand = [s for s in syllabus if skill_counts.get(s, 0) > 0]
        if not in_demand:
            return "Discontinue entirely — no syllabus skills are in demand in this district."
        elif course.placement_rate < 30:
            return "Restructure urgently — retain only in-demand modules and add emerging skills."
        else:
            return "Review and modernise — some skills still relevant but curriculum needs updating."

    @staticmethod
    def _calculate_batch_budget(skill: str, sector: str, batches: int, batch_size: int) -> Dict:
        """Estimate budget for launching new training batches."""
        total_students = batches * batch_size
        training_cost = total_students * COST_PER_STUDENT_TRAINING
        instructor_cost = batches * COURSE_DURATION_MONTHS * COST_INSTRUCTOR_PER_MONTH
        equipment_cost = SECTOR_EQUIPMENT_COST.get(sector, 100000)
        total = training_cost + instructor_cost + equipment_cost
        return {
            "training_cost": training_cost,
            "instructor_cost": instructor_cost,
            "equipment_cost": equipment_cost,
            "total_estimated_cost": total,
            "cost_per_student": round(total / max(1, total_students)),
            "currency": "INR",
        }

    @staticmethod
    def _timeline_for_action(category: str, priority: int) -> str:
        """Map action categories to realistic government timelines."""
        if priority == 1:
            return "Immediate (0-3 months)"
        elif priority == 2:
            return "Short-term (1-3 months)"
        elif priority == 3:
            return "Short-term (3-6 months)"
        elif priority == 4:
            return "Medium-term (6-12 months)"
        elif priority == 5:
            return "Ongoing (quarterly review)"
        else:
            return "Medium-term (6-12 months)"

    # ════════════════════════════════════════════════════════════════════════
    #  MAIN PLAN GENERATOR
    # ════════════════════════════════════════════════════════════════════════

    @classmethod
    def generate_district_plan(cls, db: Session, district_name: str) -> Dict:
        district = db.query(District).filter(District.name == district_name).first()
        if not district:
            raise ValueError(f"District '{district_name}' not found")

        # ─── 1. DEMAND ANALYSIS ───────────────────────────────────────────
        postings = db.query(JobPosting).filter(
            JobPosting.district_name == district_name
        ).all()

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

        total_enrolment_capacity = sum(c.enrolment_count for c in courses)
        avg_placement = round(sum(c.placement_rate for c in courses) / max(1, len(courses)), 1)

        # ─── 3. GAP ANALYSIS ──────────────────────────────────────────────
        skills_available = []
        skills_gap = []
        skills_declining = []

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

        # Oversupplied skills (taught but not demanded)
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

        # ─── 4. SECTOR-WISE GAP BREAKDOWN ────────────────────────────────
        sector_gap_breakdown: Dict[str, Dict] = {}
        for skill_entry in skills_gap:
            nsqf = skill_entry.get("nsqf_alignment") or {}
            sector = nsqf.get("sector", "Unclassified")
            if sector not in sector_gap_breakdown:
                sector_gap_breakdown[sector] = {"gap_skills": [], "gap_count": 0, "total_demand": 0}
            sector_gap_breakdown[sector]["gap_skills"].append(skill_entry["skill"])
            sector_gap_breakdown[sector]["gap_count"] += 1
            sector_gap_breakdown[sector]["total_demand"] += skill_entry["demand_count"]

        sector_gap_list = [
            {"sector": s, **data} for s, data in sorted(
                sector_gap_breakdown.items(), key=lambda x: x[1]["total_demand"], reverse=True
            )
        ]

        # ─── 5. COURSES AT RISK ───────────────────────────────────────────
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
                    "students_affected": c.enrolment_count,
                    "redirect_options": cls._find_redirect_options(db, c, courses),
                })

        # ─── 6. BATCH PLANNING WITH BUDGET ────────────────────────────────
        batch_recommendations = []
        for g in skills_gap:
            if g["demand_count"] >= HIGH_DEMAND_THRESHOLD:
                estimated_candidates = g["demand_count"] * 4
                batches_needed = max(1, (estimated_candidates + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE)
                sector = (g.get("nsqf_alignment") or {}).get("sector", "IT-ITeS")
                budget = cls._calculate_batch_budget(g["skill"], sector, batches_needed, DEFAULT_BATCH_SIZE)
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
                    "budget_estimate": budget,
                })

        # ─── 7. CURRICULUM UPDATE RECOMMENDATIONS ─────────────────────────
        curriculum_updates = []
        if skills_gap and courses:
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

        # ─── 8. EMPLOYER INTELLIGENCE ────────────────────────────────────
        employer_concerns = cls._get_employer_concerns(db, district_name)
        employer_missing_skills: Counter = Counter()
        for ec in employer_concerns:
            for ms in ec["missing_skills"]:
                employer_missing_skills[ms] += 1

        # ─── 9. GAP SCORES & URGENCY ─────────────────────────────────────
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

        # ══════════════════════════════════════════════════════════════════
        #  NEW: BEFORE vs AFTER PROJECTION
        # ══════════════════════════════════════════════════════════════════
        current_placements = sum(
            int(c.enrolment_count * c.placement_rate / 100) for c in courses
        )
        # Estimate: each batch produces batch_size candidates at ~75% placement
        projected_new_placements = sum(
            br["batches_needed"] * DEFAULT_BATCH_SIZE * 0.75 for br in batch_recommendations
        )
        # Redirecting at-risk students to better courses
        redirected_students = sum(c["students_affected"] for c in courses_at_risk)
        projected_redirect_placements = int(redirected_students * 0.65)  # 65% expected placement after redirect

        projected_total_placements = current_placements + int(projected_new_placements) + projected_redirect_placements
        placement_improvement = projected_total_placements - current_placements
        placement_improvement_pct = round(
            (placement_improvement / max(1, current_placements)) * 100, 1
        )

        # Projected gap reduction
        skills_gap_after = max(0, len(skills_gap) - len(batch_recommendations) - len(curriculum_updates))
        projected_gap_ratio = round(
            (skills_gap_after / max(1, total_demanded)) * 100, 1
        )
        gap_reduction = round(gap_ratio - projected_gap_ratio, 1)

        before_after_projection = {
            "current_state": {
                "total_skills_demanded": total_demanded,
                "skills_available": total_available,
                "skills_gap": len(skills_gap),
                "gap_ratio_percentage": round(gap_ratio, 1),
                "current_estimated_placements": current_placements,
                "courses_at_risk": len(courses_at_risk),
                "avg_placement_rate": avg_placement,
                "supply_demand_ratio": supply_demand_ratio,
                "urgency_level": urgency,
            },
            "projected_state_after_implementation": {
                "skills_gap_after": skills_gap_after,
                "projected_gap_ratio_percentage": projected_gap_ratio,
                "gap_reduction_percentage_points": gap_reduction,
                "projected_total_placements": projected_total_placements,
                "placement_improvement": placement_improvement,
                "placement_improvement_percentage": placement_improvement_pct,
                "courses_discontinued": len(courses_at_risk),
                "new_batches_launched": len(batch_recommendations),
                "curriculum_modules_added": sum(
                    len(cu["skills_to_add"]) for cu in curriculum_updates if cu["action"] == "Add Modules"
                ),
                "curriculum_modules_deprecated": sum(
                    len(cu["skills_to_remove"]) for cu in curriculum_updates if cu["action"] == "Deprecate Modules"
                ),
                "projected_urgency_level": (
                    "Moderate" if projected_gap_ratio < MODERATE_GAP_THRESHOLD
                    else "High" if projected_gap_ratio < URGENT_GAP_THRESHOLD
                    else "Critical"
                ),
            },
            "summary": (
                f"Implementing this plan reduces the skill gap from {round(gap_ratio, 1)}% to "
                f"{projected_gap_ratio}% (a {gap_reduction} percentage-point reduction), "
                f"improves estimated annual placements from {current_placements} to "
                f"{projected_total_placements} (+{placement_improvement_pct}%), "
                f"and discontinues {len(courses_at_risk)} obsolete course(s) freeing "
                f"{redirected_students} seats for redeployment."
            ),
        }

        # ══════════════════════════════════════════════════════════════════
        #  NEW: CAPACITY UTILISATION ANALYSIS
        # ══════════════════════════════════════════════════════════════════
        total_seats = sum(c.enrolment_count for c in courses)
        # Assume 80% of enrolment = actual capacity (some seats unfilled)
        actual_utilisation = int(total_seats * 0.80)  # simulated actual fill
        wasted_seats = total_seats - actual_utilisation

        # Seats locked in at-risk courses
        at_risk_seats = sum(c.enrolment_count for c in courses if
                            c.enrolment_count < LOW_ENROLMENT_THRESHOLD and
                            c.placement_rate < LOW_PLACEMENT_THRESHOLD)

        # Seats in healthy courses
        healthy_seats = sum(
            c.enrolment_count for c in courses
            if c.placement_rate >= LOW_PLACEMENT_THRESHOLD and c.enrolment_count >= LOW_ENROLMENT_THRESHOLD
        )

        # Recommended reallocation
        seats_to_reallocate = at_risk_seats + wasted_seats
        seats_needed_for_gaps = sum(
            br["batches_needed"] * DEFAULT_BATCH_SIZE for br in batch_recommendations
        )

        capacity_utilisation = {
            "total_district_capacity_seats": total_seats,
            "estimated_filled_seats": actual_utilisation,
            "utilisation_rate": round((actual_utilisation / max(1, total_seats)) * 100, 1),
            "wasted_seats": wasted_seats,
            "seats_locked_in_at_risk_courses": at_risk_seats,
            "seats_in_healthy_courses": healthy_seats,
            "seats_available_for_reallocation": seats_to_reallocate,
            "seats_needed_for_new_batches": seats_needed_for_gaps,
            "net_seat_balance": seats_to_reallocate - seats_needed_for_gaps,
            "reallocation_feasible": seats_to_reallocate >= seats_needed_for_gaps,
            "recommendation": (
                f"Reallocate {seats_to_reallocate} seats from underperforming/wasted courses "
                f"to address {seats_needed_for_gaps} seats needed for high-demand batches. "
                f"{'Feasible without new infrastructure.' if seats_to_reallocate >= seats_needed_for_gaps else 'Additional infrastructure investment needed.'}"
            ),
        }

        # ══════════════════════════════════════════════════════════════════
        #  NEW: CROSS-DISTRICT COMPARISON
        # ══════════════════════════════════════════════════════════════════
        all_districts = db.query(District).all()
        cross_district = []
        for d in all_districts:
            if d.name == district_name:
                continue
            d_postings = db.query(JobPosting).filter(JobPosting.district_name == d.name).all()
            d_courses = db.query(Course).filter(Course.district_name == d.name).all()

            d_skill_counts: Counter = Counter()
            for jp in d_postings:
                skills = json.loads(jp.extracted_skills_json) if jp.extracted_skills_json else []
                for s in skills:
                    d_skill_counts[s] += 1

            d_taught: List[str] = []
            for c in d_courses:
                syllabus = json.loads(c.syllabus_skills_json) if c.syllabus_skills_json else []
                d_taught.extend(syllabus)

            d_gaps = [
                s for s in d_skill_counts
                if not cls._is_taught(s, d_taught)
            ]

            # Overlapping gaps
            current_gap_skills = {g["skill"].lower() for g in skills_gap}
            d_gap_skills = {s.lower() for s in d_gaps}
            overlapping_gaps = list(current_gap_skills & d_gap_skills)

            d_avg_placement = round(
                sum(c.placement_rate for c in d_courses) / max(1, len(d_courses)), 1
            ) if d_courses else 0

            cross_district.append({
                "district": d.name,
                "region": d.region,
                "postings_count": len(d_postings),
                "courses_count": len(d_courses),
                "gap_count": len(d_gaps),
                "avg_placement_rate": d_avg_placement,
                "overlapping_gap_skills": overlapping_gaps,
                "shared_infrastructure_opportunity": (
                    f"{len(overlapping_gaps)} shared gaps — consider joint training facility"
                    if len(overlapping_gaps) >= 2
                    else "Minimal overlap — independent planning sufficient"
                ),
            })

        # Sort by overlap count (most shared gaps first)
        cross_district.sort(key=lambda x: len(x["overlapping_gap_skills"]), reverse=True)

        # ══════════════════════════════════════════════════════════════════
        #  NEW: EMERGING TECHNOLOGY HORIZON SCAN
        # ══════════════════════════════════════════════════════════════════
        # Check which emerging skills are NOT yet demanded but approaching
        current_demanded_skills = {s.lower() for s in skill_counts.keys()}
        current_taught_skills = {s.lower() for s in all_taught_set}

        horizon_scan = []
        for emerging in EMERGING_TECH_SKILLS:
            already_demanded = emerging["skill"].lower() in current_demanded_skills
            already_taught = emerging["skill"].lower() in current_taught_skills

            # Check if a related skill is demanded (fuzzy)
            related_demand = any(
                fuzz.token_set_ratio(emerging["skill"].lower(), ds) >= 70
                for ds in current_demanded_skills
            )

            if already_demanded:
                status = "Already in demand — prioritise training"
                priority = "High"
            elif related_demand:
                status = "Adjacent demand detected — prepare curriculum now"
                priority = "Medium"
            elif already_taught:
                status = "Already taught — monitor demand"
                priority = "Low"
            else:
                status = "Not yet demanded — monitor and prepare"
                priority = "Low"

            # Check alignment to district industries
            district_industries = json.loads(district.major_industries) if district.major_industries else []
            industry_relevant = any(
                fuzz.token_set_ratio(emerging["sector"].lower(), ind.lower()) >= 60
                for ind in district_industries
            )

            horizon_scan.append({
                "skill": emerging["skill"],
                "sector": emerging["sector"],
                "horizon": emerging["horizon"],
                "pm_setu_trade": emerging["pm_setu_trade"],
                "status": status,
                "priority": priority,
                "industry_relevance": "High" if industry_relevant else "Moderate",
                "recommendation": (
                    f"Add pilot module in next academic cycle"
                    if priority in ("High", "Medium") and industry_relevant
                    else "Monitor — not yet critical for this district"
                ),
            })

        # ══════════════════════════════════════════════════════════════════
        #  NEW: SCENARIO PLANNING
        # ══════════════════════════════════════════════════════════════════
        # Base demand by sector
        top_sectors = [s for s, _ in sector_demand.most_common(3)]
        scenarios = []

        for sector in top_sectors:
            sector_postings = sector_demand.get(sector, 0)
            # Scenario A: 20% growth
            growth_20 = int(sector_postings * 1.2)
            # Scenario B: 20% decline
            decline_20 = int(sector_postings * 0.8)

            # Find skills in this sector's gaps
            sector_gap_skills = [
                g["skill"] for g in skills_gap
                if (g.get("nsqf_alignment") or {}).get("sector") == sector
            ]

            scenarios.append({
                "sector": sector,
                "current_postings": sector_postings,
                "scenario_growth_20pct": {
                    "projected_postings": growth_20,
                    "additional_candidates_needed": (growth_20 - sector_postings) * 4,
                    "additional_batches_needed": max(0, ((growth_20 - sector_postings) * 4 + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE),
                    "impacted_gap_skills": sector_gap_skills,
                    "action": (
                        f"If {sector} grows 20%, need {max(0, ((growth_20 - sector_postings) * 4 + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE)} "
                        f"additional batch(es). Pre-emptively train instructors for: {', '.join(sector_gap_skills[:3])}."
                    ),
                },
                "scenario_decline_20pct": {
                    "projected_postings": decline_20,
                    "reduced_demand": sector_postings - decline_20,
                    "action": (
                        f"If {sector} declines 20%, reduce {max(0, ((sector_postings - decline_20) * 4 + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE)} "
                        f"batch(es). Redirect capacity to growing sectors."
                    ),
                },
            })

        # Combined scenario
        if len(top_sectors) >= 2:
            s1, s2 = top_sectors[0], top_sectors[1]
            combined_growth = int(sector_demand.get(s1, 0) * 1.2 + sector_demand.get(s2, 0) * 1.2)
            scenarios.append({
                "sector": f"{s1} + {s2} (combined growth)",
                "current_postings": sector_demand.get(s1, 0) + sector_demand.get(s2, 0),
                "scenario_combined_20pct_growth": {
                    "projected_postings": combined_growth,
                    "total_additional_candidates_needed": (combined_growth - (sector_demand.get(s1, 0) + sector_demand.get(s2, 0))) * 4,
                    "action": (
                        f"If both {s1} and {s2} grow 20%, prioritise {s1} "
                        f"(higher wage premium and skill scarcity)."
                    ),
                },
            })

        # ══════════════════════════════════════════════════════════════════
        #  10. PRIORITISED ACTION PLAN (TIME-BOUND)
        # ══════════════════════════════════════════════════════════════════
        action_items = []

        # P1: Launch critical batches
        for br in batch_recommendations:
            if br["priority"] == "High":
                action_items.append({
                    "priority": 1,
                    "category": "New Training Batch",
                    "action": (
                        f"Launch {br['batches_needed']} batch(es) of '{br['skill']}' "
                        f"({br['recommended_batch_size']} students/batch, "
                        f"{br['estimated_duration_months']} months)"
                    ),
                    "timeline": cls._timeline_for_action("New Training Batch", 1),
                    "expected_impact": (
                        f"Addresses {br['demand_count']} active job postings "
                        f"({br['demand_level']} demand). "
                        f"Estimated {int(br['batches_needed'] * br['recommended_batch_size'] * 0.75)} placements."
                    ),
                    "budget": br["budget_estimate"],
                    "nsqf_qp": br.get("nsqf_alignment", {}).get("qp_code") if br.get("nsqf_alignment") else None,
                })

        # P2: Discontinue at-risk courses
        for c in courses_at_risk:
            action_items.append({
                "priority": 2,
                "category": "Course Discontinuation",
                "action": (
                    f"Discontinue or restructure '{c['title']}' — "
                    f"redirect {c['students_affected']} students"
                ),
                "timeline": cls._timeline_for_action("Course Discontinuation", 2),
                "expected_impact": (
                    f"Frees {c['students_affected']} seats and instructor capacity. "
                    f"Redirect options: {', '.join(c['redirect_options'][:2])}."
                ),
                "budget": None,
            })

        # P3: Curriculum updates — add modules
        for cu in curriculum_updates:
            if cu["action"] == "Add Modules":
                action_items.append({
                    "priority": 3,
                    "category": "Curriculum Update",
                    "action": (
                        f"Add {', '.join(cu['skills_to_add'][:3])} modules to '{cu['target_course']}'"
                    ),
                    "timeline": cls._timeline_for_action("Curriculum Update", 3),
                    "expected_impact": "Closes demand gap without launching entirely new courses.",
                    "budget": {
                        "curriculum_development_cost": 50000,
                        "instructor_training_cost": 25000,
                        "total": 75000,
                        "currency": "INR",
                    },
                })

        # P4: Deprecate obsolete modules
        for cu in curriculum_updates:
            if cu["action"] == "Deprecate Modules":
                action_items.append({
                    "priority": 4,
                    "category": "Curriculum Cleanup",
                    "action": (
                        f"Remove obsolete modules from '{cu['target_course']}': "
                        f"{', '.join(cu['skills_to_remove'][:3])}"
                    ),
                    "timeline": cls._timeline_for_action("Curriculum Cleanup", 4),
                    "expected_impact": "Reclaims instructional hours for relevant content.",
                    "budget": None,
                })

        # P5: Employer-flagged gaps
        for skill, count in employer_missing_skills.most_common(3):
            action_items.append({
                "priority": 5,
                "category": "Employer-Flagged Gap",
                "action": f"Address employer-reported missing skill: '{skill}' (flagged by {count} employer(s))",
                "timeline": cls._timeline_for_action("Employer Gap", 5),
                "expected_impact": "Improves employer satisfaction and placement outcomes.",
                "budget": None,
            })

        # P6: Emerging tech preparation
        for es in horizon_scan:
            if es["priority"] in ("High", "Medium") and es["industry_relevance"] == "High":
                action_items.append({
                    "priority": 6,
                    "category": "Emerging Tech Preparation",
                    "action": f"Prepare pilot module for '{es['skill']}' (horizon: {es['horizon']})",
                    "timeline": cls._timeline_for_action("Emerging Tech", 6),
                    "expected_impact": f"Pre-positiones district for {es['sector']} sector growth.",
                    "budget": {
                        "pilot_module_development": 100000,
                        "instructor_upskilling": 60000,
                        "total": 160000,
                        "currency": "INR",
                    },
                })

        # P7: Monitor & validate (close the loop)
        action_items.append({
            "priority": 7,
            "category": "Monitor & Validate",
            "action": (
                "Re-scan market in 90 days. Re-assess obsolescence for flagged courses. "
                "Collect employer feedback on recommended curriculum changes. "
                "Adjust plan quarterly."
            ),
            "timeline": "Ongoing (quarterly review)",
            "expected_impact": "Closes the feedback loop — ensures continuous alignment.",
            "budget": None,
        })

        if not action_items:
            action_items.append({
                "priority": 7,
                "category": "Monitoring",
                "action": "Continue monitoring — district training supply is well aligned with current demand.",
                "timeline": "Ongoing",
                "expected_impact": "Maintain alignment",
                "budget": None,
            })

        # Sort by priority
        action_items.sort(key=lambda x: x["priority"])

        # ══════════════════════════════════════════════════════════════════
        #  FINAL ASSEMBLED PLAN
        # ══════════════════════════════════════════════════════════════════
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
                "total_enrolment_capacity": total_enrolment_capacity,
                "average_placement_rate": avg_placement,
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
                "sector_wise_gap_breakdown": sector_gap_list,
            },

            # Courses at risk
            "courses_at_risk": courses_at_risk,
            "course_health_overview": course_summary,

            # Batch plan with budget
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

            # ══ NEW: Before vs After Projection ══
            "before_after_projection": before_after_projection,

            # ══ NEW: Capacity Utilisation ══
            "capacity_utilisation": capacity_utilisation,

            # ══ NEW: Cross-District Comparison ══
            "cross_district_comparison": {
                "total_districts_compared": len(cross_district),
                "comparison": cross_district[:5],  # top 5 by overlap
                "regional_insight": (
                    f"Compared {district_name} with {len(cross_district)} other districts. "
                    f"Top overlapping gaps: {cross_district[0]['overlapping_gap_skills'][:3] if cross_district else 'None'}. "
                    f"Consider shared training infrastructure with "
                    f"{cross_district[0]['district'] if cross_district else 'N/A'} "
                    f"({len(cross_district[0]['overlapping_gap_skills']) if cross_district else 0} shared gaps)."
                ),
            },

            # ══ NEW: Emerging Technology Horizon Scan ══
            "emerging_technology_horizon_scan": {
                "total_emerging_skills_tracked": len(horizon_scan),
                "pm_setu_aligned_trades": len([h for h in horizon_scan if h["pm_setu_trade"]]),
                "high_priority_for_district": [h for h in horizon_scan if h["priority"] == "High"],
                "medium_priority_for_district": [h for h in horizon_scan if h["priority"] == "Medium"],
                "low_priority_monitoring": [h for h in horizon_scan if h["priority"] == "Low"],
                "full_scan": horizon_scan,
            },

            # ══ NEW: Scenario Planning ══
            "scenario_planning": {
                "scenarios_analysed": len(scenarios),
                "base_case": {
                    "district": district_name,
                    "top_sectors": top_sectors,
                    "total_postings": total_postings,
                },
                "scenarios": scenarios,
            },

            # Action plan (time-bound)
            "action_plan": {
                "total_actions": len(action_items),
                "urgency_level": urgency,
                "items": action_items,
                "timeline_summary": {
                    "immediate_0_3_months": len([a for a in action_items if a["priority"] in (1, 2)]),
                    "short_term_3_6_months": len([a for a in action_items if a["priority"] in (3, 4)]),
                    "medium_term_6_12_months": len([a for a in action_items if a["priority"] == 6]),
                    "ongoing_monitoring": len([a for a in action_items if a["priority"] == 7]),
                },
            },

            # Metadata
            "plan_metadata": {
                "generated_for": "SIH 26134 — Government of Maharashtra",
                "methodology": (
                    "NLP-extracted skills from job postings, fuzzy-matched against course syllabi, "
                    "weighted by demand frequency, employer feedback, and placement outcomes. "
                    "Includes before/after projection, budget estimates, capacity utilisation analysis, "
                    "cross-district comparison, emerging-tech horizon scan, and scenario planning."
                ),
                "features": [
                    "demand_supply_gap_analysis",
                    "sector_wise_breakdown",
                    "course_health_scoring",
                    "batch_planning_with_budget",
                    "before_after_projection",
                    "capacity_utilisation",
                    "cross_district_comparison",
                    "emerging_tech_horizon_scan",
                    "scenario_planning",
                    "time_bound_action_plan",
                    "employer_feedback_loop",
                    "nsqf_alignment",
                ],
            },
        }

    # ════════════════════════════════════════════════════════════════════════
    #  HELPER: Find redirect options for discontinued courses
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _find_redirect_options(db: Session, at_risk_course: Course, all_courses: List[Course]) -> List[str]:
        """Find better courses to redirect students from a discontinued course."""
        # Find courses in the same district with good placement
        candidates = [
            c for c in all_courses
            if c.id != at_risk_course.id
            and c.placement_rate >= 60.0
            and c.enrolment_count < 200  # has capacity
        ]
        candidates.sort(key=lambda c: c.placement_rate, reverse=True)
        return [f"{c.title} ({c.placement_rate}% placement)" for c in candidates[:3]]
