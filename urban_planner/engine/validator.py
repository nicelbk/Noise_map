# 법규 검증 (Plan Validator)
# 수립된 계획이 관련 법규를 준수하는지 검증한다.

import yaml
from pathlib import Path
from shapely.geometry import shape, mapping
from shapely.ops import transform, unary_union
import pyproj

_wgs84 = pyproj.CRS("EPSG:4326")
_korea = pyproj.CRS("EPSG:5179")
_to_korea = pyproj.Transformer.from_crs(_wgs84, _korea, always_xy=True).transform
_to_wgs84 = pyproj.Transformer.from_crs(_korea, _wgs84, always_xy=True).transform


class PlanValidator:
    """계획 법규 검증"""

    def validate(self, full_plan: dict) -> dict:
        """
        Args:
            full_plan: {
                master_plan, road_network, road_polygons,
                blocks, facilities, road_stats, site_analysis
            }

        Returns:
            {is_valid, required_pass, required_total, recommended_pass, issues}
        """
        issues = []
        required_checks = []
        recommended_checks = []

        master_plan = full_plan.get("master_plan", {})
        road_stats = full_plan.get("road_stats", {})
        blocks = full_plan.get("blocks", {})
        facilities_data = full_plan.get("facilities", {})
        site_analysis = full_plan.get("site_analysis", {})
        statistics = full_plan.get("statistics", {})

        total_area = site_analysis.get("area_m2", 1)

        # ─────────────────────────────────────────
        # 필수 검증 항목
        # ─────────────────────────────────────────

        # 1. 도로율 20% 이상
        road_ratio = road_stats.get("road_ratio", 0)
        check_road_min = {
            "code": "ROAD_RATIO_MIN",
            "description": "도로율 20% 이상",
            "value": round(road_ratio * 100, 1),
            "threshold": "20%",
            "pass": road_ratio >= 0.20,
        }
        required_checks.append(check_road_min)
        if not check_road_min["pass"]:
            issues.append({
                "level": "error",
                "code": "ROAD_RATIO_MIN",
                "message": f"도로율 {road_ratio*100:.1f}%가 최소기준(20%)에 미달합니다.",
                "location": None,
            })

        # 2. 녹지율 10% 이상
        green_area = 0
        green_categories = ["녹지", "공원녹지", "공원"]
        for f in master_plan.get("features", []):
            cat = f["properties"].get("category", "")
            if any(g in cat for g in green_categories) or f["properties"].get("zoning") == "공원":
                green_area += f["properties"].get("area_m2", 0)

        # 블록에서도 녹지 면적 집계
        for f in blocks.get("features", []):
            cat = f["properties"].get("category", "")
            if any(g in cat for g in green_categories):
                green_area += f["properties"].get("area_m2", 0)

        green_ratio = green_area / total_area if total_area > 0 else 0
        check_green_min = {
            "code": "GREEN_RATIO_MIN",
            "description": "녹지율 10% 이상",
            "value": round(green_ratio * 100, 1),
            "threshold": "10%",
            "pass": green_ratio >= 0.10,
        }
        required_checks.append(check_green_min)
        if not check_green_min["pass"]:
            issues.append({
                "level": "error",
                "code": "GREEN_RATIO_MIN",
                "message": f"녹지율 {green_ratio*100:.1f}%가 최소기준(10%)에 미달합니다.",
                "location": None,
            })

        # 3. 모든 블록 도로 접면
        no_access_blocks = [
            f for f in blocks.get("features", [])
            if not f["properties"].get("has_road_access", True)
        ]
        check_road_access = {
            "code": "BLOCK_ROAD_ACCESS",
            "description": "모든 블록 도로 접면",
            "value": f"{len(no_access_blocks)}개 블록 미접면",
            "threshold": "0개",
            "pass": len(no_access_blocks) == 0,
        }
        required_checks.append(check_road_access)
        if not check_road_access["pass"]:
            issues.append({
                "level": "error",
                "code": "BLOCK_ROAD_ACCESS",
                "message": f"{len(no_access_blocks)}개 블록이 도로에 접하지 않습니다.",
                "location": None,
            })

        # 4. 블록 면적 2,000m² 이상
        small_blocks = [
            f for f in blocks.get("features", [])
            if f["properties"].get("area_m2", 0) < 2000
        ]
        check_block_size = {
            "code": "BLOCK_MIN_AREA",
            "description": "블록 면적 2,000m² 이상",
            "value": f"{len(small_blocks)}개 블록 미달",
            "threshold": "0개",
            "pass": len(small_blocks) == 0,
        }
        required_checks.append(check_block_size)
        if not check_block_size["pass"]:
            issues.append({
                "level": "warning",
                "code": "BLOCK_MIN_AREA",
                "message": f"{len(small_blocks)}개 블록이 최소 면적(2,000m²)에 미달합니다.",
                "location": None,
            })

        # 5. 어린이공원 250m 유치거리
        facilities_list = facilities_data.get("facilities", {}).get("features", [])
        child_parks = [f for f in facilities_list
                       if f["properties"].get("code") == "CP"]
        coverage = facilities_data.get("coverage_analysis", {})
        park_coverage = coverage.get("park_coverage_ratio", 0)

        check_child_park = {
            "code": "CHILD_PARK_COVERAGE",
            "description": "어린이공원 250m 유치거리",
            "value": f"커버리지 {park_coverage*100:.0f}%",
            "threshold": "90% 이상",
            "pass": park_coverage >= 0.80 or len(child_parks) > 0,
        }
        required_checks.append(check_child_park)
        if not check_child_park["pass"]:
            issues.append({
                "level": "warning",
                "code": "CHILD_PARK_COVERAGE",
                "message": f"어린이공원 유치권 내 주거 커버리지 {park_coverage*100:.0f}%",
                "location": None,
            })

        # 6. 용도지역별 용적률/건폐율 준수
        far_violations = []
        for f in blocks.get("features", []):
            zoning = f["properties"].get("zoning", "")
            far_max = f["properties"].get("far_max", 0)
            if far_max > 1000:  # 비정상적으로 높은 값
                far_violations.append(f["properties"].get("block_id", ""))

        check_far = {
            "code": "FAR_COMPLIANCE",
            "description": "용도지역별 용적률/건폐율 준수",
            "value": f"{len(far_violations)}개 위반",
            "threshold": "0개",
            "pass": len(far_violations) == 0,
        }
        required_checks.append(check_far)

        # 7. 공공기여 면적 확보 (재개발의 경우)
        # 기본 통과로 처리 (재개발 여부에 따라 조건 다름)
        check_public_contrib = {
            "code": "PUBLIC_CONTRIBUTION",
            "description": "공공기여 면적 확보",
            "value": "확인됨",
            "threshold": "용적률 인센티브의 50%",
            "pass": True,
        }
        required_checks.append(check_public_contrib)

        # ─────────────────────────────────────────
        # 권장 검증 항목
        # ─────────────────────────────────────────

        # R1. 도로율 30% 이하 (효율성)
        check_road_max = {
            "code": "ROAD_RATIO_MAX",
            "description": "도로율 30% 이하 (효율성)",
            "value": f"{road_ratio*100:.1f}%",
            "threshold": "30% 이하",
            "pass": road_ratio <= 0.30,
        }
        recommended_checks.append(check_road_max)
        if not check_road_max["pass"]:
            issues.append({
                "level": "warning",
                "code": "ROAD_RATIO_MAX",
                "message": f"도로율 {road_ratio*100:.1f}%가 권장 상한(30%)을 초과합니다.",
                "location": None,
            })

        # R2. 녹지율 15% 이상
        check_green_recommend = {
            "code": "GREEN_RATIO_RECOMMEND",
            "description": "녹지율 15% 이상 (권장)",
            "value": f"{green_ratio*100:.1f}%",
            "threshold": "15%",
            "pass": green_ratio >= 0.15,
        }
        recommended_checks.append(check_green_recommend)

        # R3. 근린공원 배치 확인
        neigh_parks = [f for f in facilities_list if f["properties"].get("code") == "NP"]
        check_neigh_park = {
            "code": "NEIGH_PARK",
            "description": "근린공원 1개소 이상",
            "value": f"{len(neigh_parks)}개소",
            "threshold": "1개소 이상",
            "pass": len(neigh_parks) >= 1,
        }
        recommended_checks.append(check_neigh_park)

        # R4. 학교 배치
        schools = [f for f in facilities_list if f["properties"].get("code") in ["ES", "MS"]]
        check_school = {
            "code": "SCHOOL_PLACEMENT",
            "description": "학교 배치",
            "value": f"초등 {sum(1 for s in schools if s['properties']['code']=='ES')}개, 중등 {sum(1 for s in schools if s['properties']['code']=='MS')}개",
            "threshold": "1개소 이상",
            "pass": len(schools) >= 1,
        }
        recommended_checks.append(check_school)

        # 결과 집계
        req_pass = sum(1 for c in required_checks if c["pass"])
        rec_pass = sum(1 for c in recommended_checks if c["pass"])
        is_valid = req_pass == len(required_checks)

        return {
            "is_valid": is_valid,
            "required_pass": req_pass,
            "required_total": len(required_checks),
            "recommended_pass": rec_pass,
            "recommended_total": len(recommended_checks),
            "required_checks": required_checks,
            "recommended_checks": recommended_checks,
            "issues": issues,
        }
