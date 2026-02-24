# 결과 시각화 (Visualizer)
# 계획 결과를 이미지로 내보내는 기능

import io
import base64
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 헤드리스 모드
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

import numpy as np
from shapely.geometry import shape
from shapely.ops import transform
import pyproj

_wgs84 = pyproj.CRS("EPSG:4326")
_korea = pyproj.CRS("EPSG:5179")
_to_korea = pyproj.Transformer.from_crs(_wgs84, _korea, always_xy=True).transform


# 용도지역 색상 맵
ZONING_COLORS = {
    "제1종전용주거": "#FFE4E1",
    "제2종전용주거": "#FFD0C8",
    "제1종일반주거": "#FFB6C1",
    "제2종일반주거": "#FF91A4",
    "제3종일반주거": "#FF69B4",
    "준주거":        "#DDA0DD",
    "중심상업":      "#FF4500",
    "일반상업":      "#FF6347",
    "근린상업":      "#FF8C00",
    "유통상업":      "#FFA500",
    "전용공업":      "#808080",
    "일반공업":      "#A9A9A9",
    "준공업":        "#C0C0C0",
    "보전녹지":      "#228B22",
    "생산녹지":      "#32CD32",
    "자연녹지":      "#7CFC00",
    "공원":          "#90EE90",
    "공공시설":      "#4169E1",
    "도로":          "#AAAAAA",
}

FACILITY_COLORS = {
    "근린공원":   "#32CD32",
    "어린이공원": "#90EE90",
    "초등학교":   "#4169E1",
    "중학교":     "#1E90FF",
    "주민센터":   "#9370DB",
}


class Visualizer:
    """계획 결과 시각화"""

    def render_plan(self, full_plan: dict, dpi: int = 150) -> str:
        """
        전체 계획을 PNG로 렌더링하고 base64 인코딩 반환

        Args:
            full_plan: 전체 계획 데이터
            dpi: 이미지 해상도

        Returns:
            base64 인코딩된 PNG 문자열
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect("equal")
        ax.set_facecolor("#F0F0F0")

        # 대상지 경계
        if "boundary_geojson" in full_plan:
            self._draw_boundary(ax, full_plan["boundary_geojson"])

        # 마스터플랜 (용도지역)
        if "master_plan" in full_plan:
            self._draw_master_plan(ax, full_plan["master_plan"])

        # 도로망
        if "road_polygons" in full_plan:
            self._draw_roads(ax, full_plan["road_polygons"])

        # 공공시설
        if "facilities" in full_plan:
            self._draw_facilities(ax, full_plan["facilities"])

        # 범례
        self._add_legend(ax)

        # 제목
        site_name = full_plan.get("site_name", "대상지")
        ax.set_title(f"토지이용계획 - {site_name}", fontsize=14, fontweight="bold", pad=10)

        plt.tight_layout()

        # PNG → base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return img_base64

    def _draw_boundary(self, ax, boundary_geojson: dict):
        """대상지 경계 그리기"""
        try:
            if boundary_geojson.get("type") == "FeatureCollection":
                geom_wgs84 = shape(boundary_geojson["features"][0]["geometry"])
            elif boundary_geojson.get("type") == "Feature":
                geom_wgs84 = shape(boundary_geojson["geometry"])
            else:
                geom_wgs84 = shape(boundary_geojson)

            geom_korea = transform(_to_korea, geom_wgs84)
            x, y = geom_korea.exterior.xy
            ax.plot(x, y, "k-", linewidth=2, zorder=10)
        except Exception:
            pass

    def _draw_master_plan(self, ax, master_plan: dict):
        """용도지역 폴리곤 그리기"""
        for feature in master_plan.get("features", []):
            try:
                geom_wgs84 = shape(feature["geometry"])
                geom_korea = transform(_to_korea, geom_wgs84)
                zoning = feature["properties"].get("zoning", "")
                color = ZONING_COLORS.get(zoning, "#CCCCCC")

                if geom_korea.geom_type == "Polygon":
                    x, y = geom_korea.exterior.xy
                    ax.fill(x, y, color=color, alpha=0.7, zorder=2)
                    ax.plot(x, y, "-", color="#666666", linewidth=0.5, zorder=3)
                elif geom_korea.geom_type == "MultiPolygon":
                    for poly in geom_korea.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, color=color, alpha=0.7, zorder=2)
                        ax.plot(x, y, "-", color="#666666", linewidth=0.5, zorder=3)
            except Exception:
                pass

    def _draw_roads(self, ax, road_polygons: dict):
        """도로 폴리곤 그리기"""
        for feature in road_polygons.get("features", []):
            try:
                geom_wgs84 = shape(feature["geometry"])
                geom_korea = transform(_to_korea, geom_wgs84)

                if geom_korea.geom_type == "Polygon":
                    x, y = geom_korea.exterior.xy
                    ax.fill(x, y, color="#AAAAAA", alpha=0.9, zorder=4)
                elif geom_korea.geom_type == "MultiPolygon":
                    for poly in geom_korea.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, color="#AAAAAA", alpha=0.9, zorder=4)
            except Exception:
                pass

    def _draw_facilities(self, ax, facilities: dict):
        """공공시설 그리기"""
        for feature in facilities.get("features", []):
            try:
                geom_wgs84 = shape(feature["geometry"])
                geom_korea = transform(_to_korea, geom_wgs84)
                fac_type = feature["properties"].get("facility_type", "")
                color = FACILITY_COLORS.get(fac_type, "#4169E1")

                if geom_korea.geom_type == "Polygon":
                    x, y = geom_korea.exterior.xy
                    ax.fill(x, y, color=color, alpha=0.85, zorder=6)
                    ax.plot(x, y, "-", color="white", linewidth=1, zorder=7)

                # 시설 레이블
                centroid = geom_korea.centroid
                label = feature["properties"].get("label", fac_type)
                ax.annotate(
                    label,
                    xy=(centroid.x, centroid.y),
                    ha="center", va="center",
                    fontsize=5, color="white", fontweight="bold",
                    zorder=8,
                )
            except Exception:
                pass

    def _add_legend(self, ax):
        """범례 추가"""
        legend_items = [
            mpatches.Patch(color="#FFB6C1", label="주거지역(1종)"),
            mpatches.Patch(color="#FF69B4", label="주거지역(3종)"),
            mpatches.Patch(color="#FF8C00", label="상업지역"),
            mpatches.Patch(color="#90EE90", label="공원/녹지"),
            mpatches.Patch(color="#4169E1", label="공공시설"),
            mpatches.Patch(color="#AAAAAA", label="도로"),
        ]
        ax.legend(
            handles=legend_items,
            loc="lower right",
            fontsize=7,
            framealpha=0.9,
        )

    def generate_statistics_report(self, full_plan: dict) -> str:
        """통계 리포트 텍스트 생성"""
        stats = full_plan.get("statistics", {})
        site_analysis = full_plan.get("site_analysis", {})
        road_stats = full_plan.get("road_stats", {})
        facilities_data = full_plan.get("facilities", {})
        validation = full_plan.get("validation", {})

        area_m2 = site_analysis.get("area_m2", 0)
        area_ha = site_analysis.get("area_ha", 0)

        estimates = facilities_data.get("estimates", {})
        households = estimates.get("households", 0)
        population = estimates.get("population", 0)

        facilities_list = facilities_data.get("facilities", {}).get("features", [])

        report_lines = [
            "=" * 45,
            "   토지이용계획 통계",
            "=" * 45,
            f"대상지 면적: {area_m2:,.0f} m² ({area_ha:.1f} ha)",
            "",
            "[ 용도별 면적 ]",
        ]

        # 용도별 면적
        category_map = {
            "주거": "주거지역",
            "상업": "상업지역",
            "공업": "공업지역",
            "녹지": "녹지/공원",
            "공원녹지": "녹지/공원",
            "공원": "공원",
            "공공": "공공시설",
            "도로": "도로",
        }
        area_by_cat = {}
        for f in full_plan.get("blocks", {}).get("features", []):
            cat = f["properties"].get("category", "기타")
            a = f["properties"].get("area_m2", 0)
            area_by_cat[cat] = area_by_cat.get(cat, 0) + a

        for cat, label in [
            ("주거", "주거지역"), ("상업", "상업지역"), ("공업", "공업지역"),
            ("녹지", "녹지/공원"), ("공원녹지", "녹지/공원"), ("공원", "공원"),
            ("공공", "공공시설"),
        ]:
            a = area_by_cat.get(cat, 0)
            if a > 0:
                pct = a / area_m2 * 100 if area_m2 > 0 else 0
                report_lines.append(f"  {label:<10}: {a:>10,.0f} m² ({pct:.1f}%)")

        road_area = road_stats.get("road_area_m2", 0)
        road_pct = road_area / area_m2 * 100 if area_m2 > 0 else 0
        report_lines.append(f"  {'도로':<10}: {road_area:>10,.0f} m² ({road_pct:.1f}%)")

        report_lines += [
            "",
            "[ 수용 능력 추정 ]",
            f"  예상 세대수: {households:>8,} 세대",
            f"  예상 인구  : {population:>8,} 명",
            "",
            "[ 도로망 ]",
            f"  총 도로 연장: {road_stats.get('total_length_m', 0):>8,.0f} m",
            f"  도로율      : {road_stats.get('road_ratio', 0)*100:>8.1f} %",
            "",
            "[ 공공시설 ]",
        ]

        # 시설 집계
        cp_cnt = sum(1 for f in facilities_list if f["properties"].get("code") == "CP")
        np_cnt = sum(1 for f in facilities_list if f["properties"].get("code") == "NP")
        es_cnt = sum(1 for f in facilities_list if f["properties"].get("code") == "ES")
        ms_cnt = sum(1 for f in facilities_list if f["properties"].get("code") == "MS")
        cc_cnt = sum(1 for f in facilities_list if f["properties"].get("code") == "CC")

        coverage = facilities_data.get("coverage_analysis", {})
        park_cov = coverage.get("park_coverage_ratio", 0) * 100

        report_lines += [
            f"  어린이공원: {cp_cnt}개소 (유치권 커버 {park_cov:.0f}%)",
            f"  근린공원  : {np_cnt}개소",
            f"  학교      : 초등 {es_cnt}, 중등 {ms_cnt}",
            f"  주민센터  : {cc_cnt}개소",
            "",
            "[ 법규 검증 ]",
            f"  필수기준: {validation.get('required_pass', 0)}/{validation.get('required_total', 0)} 통과",
            f"  권장기준: {validation.get('recommended_pass', 0)}/{validation.get('recommended_total', 0)} 통과",
            "=" * 45,
        ]

        return "\n".join(report_lines)
