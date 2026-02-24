"""
토지이용계획 수립 모듈 (용도지역 배치)
"""

import os
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from pyproj import Transformer

EPSG_WGS84 = 4326
EPSG_5179 = 5179

transformer_to_5179 = Transformer.from_crs(EPSG_WGS84, EPSG_5179, always_xy=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 용도 코드별 표준 정보
ZONING_INFO = {
    'R1': {'name': '제1종전용주거', 'far': 100, 'color': '#FFE4E1'},
    'R2': {'name': '제2종전용주거', 'far': 120, 'color': '#FFCDD2'},
    'R3': {'name': '제1종일반주거', 'far': 150, 'color': '#FFB3BA'},
    'R4': {'name': '제2종일반주거', 'far': 200, 'color': '#FF8FAB'},
    'R5': {'name': '제3종일반주거', 'far': 300, 'color': '#FF69B4'},
    'RQ': {'name': '준주거', 'far': 500, 'color': '#DDA0DD'},
    'CB': {'name': '중심상업', 'far': 1000, 'color': '#FF2400'},
    'CG': {'name': '일반상업', 'far': 800, 'color': '#FF4500'},
    'CN': {'name': '근린상업', 'far': 600, 'color': '#FF8C00'},
    'IQ': {'name': '준공업', 'far': 400, 'color': '#A9A9A9'},
    'GN': {'name': '자연녹지', 'far': 100, 'color': '#228B22'},
    'PK': {'name': '공원', 'far': 0, 'color': '#90EE90'},
    'PF': {'name': '공공시설', 'far': 200, 'color': '#4169E1'},
    'ROAD': {'name': '도로', 'far': 0, 'color': '#CCCCCC'},
}


class MasterPlanner:

    def plan(
        self,
        boundary_geojson: dict,
        site_analysis: dict,
        design_params: dict,
        context_analysis: dict
    ) -> dict:
        """
        반환:
        {
          'zoning_gdf': GeoDataFrame,    # 용도지역 Polygon들
          'zoning_geojson': dict,        # WGS84 GeoJSON
          'zoning_stats': dict,          # 용도별 면적/비율
          'design_rationale': str        # 배치 근거 텍스트
        }
        """
        try:
            area_ha = site_analysis.get('area_ha', 10.0)
            similar_cases = site_analysis.get('similar_cases', [])

            # Calculate target zoning ratios
            target_ratios = self._calculate_zoning_ratios(
                area_ha, design_params, context_analysis, similar_cases
            )

            # Get blocks GDF from design_params (passed from block_divider)
            # If blocks not available, create a simple division from boundary
            blocks_gdf = design_params.get('_blocks_gdf', None)
            if blocks_gdf is None or (hasattr(blocks_gdf, 'empty') and blocks_gdf.empty):
                blocks_gdf = self._create_simple_blocks(boundary_geojson, area_ha)

            # Assign zoning to blocks
            zoning_gdf = self._assign_zoning_to_blocks(
                blocks_gdf, target_ratios, context_analysis
            )

            # Smooth boundaries
            zoning_gdf = self._smooth_zoning_boundaries(zoning_gdf)

            # Calculate stats
            total_area = zoning_gdf.geometry.area.sum()
            zoning_stats = {}
            for code in zoning_gdf['zone_code'].unique():
                mask = zoning_gdf['zone_code'] == code
                zone_area = zoning_gdf[mask].geometry.area.sum()
                info = ZONING_INFO.get(code, {'name': code, 'far': 0})
                zoning_stats[code] = {
                    'name': info['name'],
                    'area_m2': round(float(zone_area), 1),
                    'area_ha': round(float(zone_area) / 10000, 3),
                    'ratio': round(float(zone_area) / total_area, 4) if total_area > 0 else 0,
                    'far_max': info.get('far', 0),
                }

            # Convert to WGS84 GeoJSON
            zoning_wgs84 = zoning_gdf.to_crs(EPSG_WGS84)
            zoning_geojson = json.loads(zoning_wgs84.to_json())

            # Generate design rationale text
            design_rationale = self._generate_rationale(
                zoning_stats, context_analysis, design_params, similar_cases
            )

            return {
                'zoning_gdf': zoning_gdf,
                'zoning_geojson': zoning_geojson,
                'zoning_stats': zoning_stats,
                'design_rationale': design_rationale,
            }

        except Exception as e:
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{EPSG_5179}")
            return {
                'zoning_gdf': empty_gdf,
                'zoning_geojson': {'type': 'FeatureCollection', 'features': []},
                'zoning_stats': {},
                'design_rationale': f'계획 수립 중 오류 발생: {str(e)}',
            }

    def _calculate_zoning_ratios(
        self,
        area_ha: float,
        design_params: dict,
        context_analysis: dict,
        similar_cases: list
    ) -> dict:
        """
        용도별 목표 면적 비율 계산
        반환: {'residential': 0.5, 'commercial': 0.1, ...}
        """
        program = design_params.get('program', '혼합용도')
        green_ratio_min = design_params.get('green_ratio_min', 0.10)
        road_ratio = design_params.get('road_ratio_target', 0.25)
        is_tod = context_analysis.get('is_tod_zone', False)

        # Priority weights
        priority = design_params.get('priority', {})
        density = priority.get('density', 3)
        green = priority.get('green', 3)
        publicfacility = priority.get('publicfacility', 3)

        # Base ratios by program
        if program == '주거중심':
            base = {
                'residential': 0.55,
                'commercial': 0.06,
                'green': 0.18,
                'public': 0.09,
                'road': road_ratio,
            }
        elif program == '상업중심':
            base = {
                'residential': 0.30,
                'commercial': 0.22,
                'green': 0.15,
                'public': 0.08,
                'road': road_ratio,
            }
        elif program == '자족도시':
            base = {
                'residential': 0.38,
                'commercial': 0.10,
                'industrial': 0.07,
                'green': 0.20,
                'public': 0.10,
                'road': road_ratio,
            }
        else:  # 혼합용도 (default)
            base = {
                'residential': 0.45,
                'commercial': 0.10,
                'green': 0.18,
                'public': 0.10,
                'road': road_ratio,
            }

        # Adjust for TOD zone
        if is_tod:
            base['residential'] = min(base['residential'] + 0.05, 0.65)
            base['commercial'] = min(base['commercial'] + 0.03, 0.25)

        # Adjust for green priority
        green_adj = (green - 3) * 0.02
        base['green'] = max(green_ratio_min, base['green'] + green_adj)

        # Adjust for public facility priority
        pf_adj = (publicfacility - 3) * 0.01
        base['public'] = max(0.05, base.get('public', 0.08) + pf_adj)

        # Adjust for density priority (affects residential type mix later)
        # Higher density → more R4/R5/RQ

        # Normalize to sum to 1.0 (excluding road which is fixed)
        non_road_sum = sum(v for k, v in base.items() if k != 'road')
        if non_road_sum > 0:
            scale = (1.0 - base['road']) / non_road_sum
            for k in list(base.keys()):
                if k != 'road':
                    base[k] = base[k] * scale

        # Incorporate similar cases (weighted average)
        if similar_cases:
            best_case = similar_cases[0]
            sim_score = best_case.get('similarity_score', 0)
            if sim_score > 0.5:
                case_mix = best_case.get('zoning_mix', {})
                blend = sim_score * 0.3  # up to 30% blend with best case
                if 'residential' in case_mix:
                    base['residential'] = (1 - blend) * base.get('residential', 0.45) + blend * case_mix['residential']
                if 'commercial' in case_mix:
                    base['commercial'] = (1 - blend) * base.get('commercial', 0.10) + blend * case_mix['commercial']
                if 'green' in case_mix:
                    green_from_case = case_mix['green']
                    base['green'] = max(green_ratio_min, (1 - blend) * base.get('green', 0.18) + blend * green_from_case)

        return base

    def _assign_zoning_to_blocks(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        target_ratios: dict,
        context_analysis: dict
    ) -> gpd.GeoDataFrame:
        """
        각 블록에 용도지역 코드 부여
        """
        if blocks_gdf.empty:
            return gpd.GeoDataFrame(
                columns=['geometry', 'zone_code', 'zone_name', 'area_m2'],
                crs=f"EPSG:{EPSG_5179}"
            )

        gdf = blocks_gdf.copy()
        total_area = gdf.geometry.area.sum()
        is_tod = context_analysis.get('is_tod_zone', False)
        nearest_subway_m = context_analysis.get('nearest_subway_m', 9999)

        # Sort blocks by centroid Y (north = high Y in EPSG:5179)
        gdf['_cy'] = gdf.geometry.centroid.y
        gdf['_cx'] = gdf.geometry.centroid.x
        gdf['_area'] = gdf.geometry.area

        max_cy = gdf['_cy'].max()
        min_cy = gdf['_cy'].min()
        max_cx = gdf['_cx'].max()
        min_cx = gdf['_cx'].min()
        cy_range = max(max_cy - min_cy, 1)
        cx_range = max(max_cx - min_cx, 1)

        # Centroid coordinates
        centroid_x = gdf['_cx'].mean()
        centroid_y = gdf['_cy'].mean()

        def assign_zone(row):
            cy = row['_cy']
            cx = row['_cx']
            area = row['_area']
            adj_road_w = row.get('adjacent_road_width_m', 0)
            is_corner = row.get('is_corner', False)

            # Normalized positions
            north_ratio = (cy - min_cy) / cy_range  # 0=south, 1=north
            east_ratio = (cx - min_cx) / cx_range   # 0=west, 1=east

            # Distance from site centroid
            dist_from_center = ((cx - centroid_x)**2 + (cy - centroid_y)**2)**0.5

            # Rule 1: Widest road frontage → commercial
            if adj_road_w >= 20 and is_corner:
                return 'CB'  # 중심상업
            if adj_road_w >= 15:
                return 'CG'  # 일반상업
            if adj_road_w >= 12:
                return 'CN'  # 근린상업

            # Rule 2: TOD zone → high-density residential
            if is_tod and nearest_subway_m < 300:
                return 'RQ'  # 준주거
            if is_tod and nearest_subway_m < 500:
                return 'R5'  # 제3종일반주거

            # Rule 3: North blocks → residential (일조)
            if north_ratio > 0.75:
                if area > 8000:
                    return 'R4'
                return 'R3'

            # Rule 4: Central large blocks
            if area > 12000 and dist_from_center < cy_range * 0.3:
                return 'R5'

            # Rule 5: Small blocks near periphery
            if area < 3000:
                return 'CN'

            # Default: second/third general residential
            if north_ratio > 0.5:
                return 'R3'
            else:
                return 'R4'

        gdf['zone_code'] = gdf.apply(assign_zone, axis=1)

        # Now enforce target ratios by adjusting assignments
        gdf = self._enforce_target_ratios(gdf, target_ratios, total_area, is_tod)

        gdf['zone_name'] = gdf['zone_code'].map(
            lambda c: ZONING_INFO.get(c, {}).get('name', c)
        )
        gdf['area_m2'] = gdf.geometry.area

        return gdf[['geometry', 'zone_code', 'zone_name', 'area_m2']].copy()

    def _enforce_target_ratios(
        self,
        gdf: gpd.GeoDataFrame,
        target_ratios: dict,
        total_area: float,
        is_tod: bool
    ) -> gpd.GeoDataFrame:
        """목표 비율에 맞게 용도 조정"""
        target_residential = target_ratios.get('residential', 0.45) * total_area
        target_commercial = target_ratios.get('commercial', 0.10) * total_area
        target_green = target_ratios.get('green', 0.18) * total_area
        target_public = target_ratios.get('public', 0.10) * total_area

        # Current assigned areas
        def get_area(codes):
            mask = gdf['zone_code'].isin(codes)
            return gdf[mask].geometry.area.sum()

        residential_codes = ['R1', 'R2', 'R3', 'R4', 'R5', 'RQ']
        commercial_codes = ['CB', 'CG', 'CN']

        curr_res = get_area(residential_codes)
        curr_com = get_area(commercial_codes)

        # If commercial is too high, convert some to residential
        if curr_com > target_commercial * 1.3:
            cn_idx = gdf[gdf['zone_code'] == 'CN'].sort_values('_area').index
            com_excess = curr_com - target_commercial
            cum = 0
            for idx in cn_idx:
                if cum >= com_excess:
                    break
                gdf.loc[idx, 'zone_code'] = 'R4'
                cum += gdf.loc[idx, '_area']

        # Assign green space (largest park-suitable blocks)
        green_needed = target_green
        if green_needed > 0:
            # Pick largest blocks that are currently R3 or R4 for green
            candidates = gdf[gdf['zone_code'].isin(['R3', 'R4', 'R5'])].sort_values(
                '_area', ascending=False
            )
            cum_green = 0
            green_target_area = min(green_needed, total_area * 0.25)
            for idx, row in candidates.iterrows():
                if cum_green >= green_target_area:
                    break
                # Pick a few large blocks for parks
                if row['_area'] > 3000 and cum_green < green_target_area * 0.5:
                    gdf.loc[idx, 'zone_code'] = 'GN'
                    cum_green += row['_area']

        # Assign public facilities
        pf_needed = target_public
        if pf_needed > 0:
            pf_candidates = gdf[gdf['zone_code'].isin(['R3', 'R4'])].sort_values('_area')
            cum_pf = 0
            for idx, row in pf_candidates.iterrows():
                if cum_pf >= pf_needed:
                    break
                if 2000 <= row['_area'] <= 10000:
                    gdf.loc[idx, 'zone_code'] = 'PF'
                    cum_pf += row['_area']

        return gdf

    def _smooth_zoning_boundaries(
        self,
        zoning_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        동일 용도 인접 블록 합치기 (dissolve)
        경계선 simplify(tolerance=1.5m)
        """
        try:
            if zoning_gdf.empty:
                return zoning_gdf

            dissolved = zoning_gdf.dissolve(by='zone_code', aggfunc='first').reset_index()
            dissolved['geometry'] = dissolved['geometry'].simplify(tolerance=1.5, preserve_topology=True)
            dissolved['area_m2'] = dissolved.geometry.area
            dissolved['zone_name'] = dissolved['zone_code'].map(
                lambda c: ZONING_INFO.get(c, {}).get('name', c)
            )
            return dissolved
        except Exception:
            return zoning_gdf

    def _create_simple_blocks(
        self,
        boundary_geojson: dict,
        area_ha: float
    ) -> gpd.GeoDataFrame:
        """블록이 없을 때 경계를 단순 분할"""
        geom = shape(boundary_geojson)
        coords = list(geom.exterior.coords)
        transformed = [transformer_to_5179.transform(lon, lat) for lon, lat in coords]
        poly = Polygon(transformed)

        # Simple 4-quadrant division
        minx, miny, maxx, maxy = poly.bounds
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        blocks = []
        quadrants = [
            Polygon([(minx, cy), (cx, cy), (cx, maxy), (minx, maxy)]),  # NW
            Polygon([(cx, cy), (maxx, cy), (maxx, maxy), (cx, maxy)]),  # NE
            Polygon([(minx, miny), (cx, miny), (cx, cy), (minx, cy)]),  # SW
            Polygon([(cx, miny), (maxx, miny), (maxx, cy), (cx, cy)]),  # SE
        ]

        for i, q in enumerate(quadrants):
            intersection = poly.intersection(q)
            if not intersection.is_empty and intersection.area > 0:
                blocks.append({
                    'geometry': intersection,
                    'block_id': f'B-{i+1:03d}',
                    'area_m2': intersection.area,
                    'adjacent_road_width_m': 0.0,
                    'adjacent_road_count': 0,
                    'is_corner': False,
                    'centroid_x': intersection.centroid.x,
                    'centroid_y': intersection.centroid.y,
                })

        return gpd.GeoDataFrame(blocks, crs=f"EPSG:{EPSG_5179}")

    def _generate_rationale(
        self,
        zoning_stats: dict,
        context_analysis: dict,
        design_params: dict,
        similar_cases: list
    ) -> str:
        """설계 근거 텍스트 자동 생성"""
        program = design_params.get('program', '혼합용도')
        is_tod = context_analysis.get('is_tod_zone', False)
        subway_name = context_analysis.get('nearest_subway_name', '없음')
        subway_dist = context_analysis.get('nearest_subway_m', 9999)

        lines = []
        lines.append(f"## 토지이용계획 배치 근거\n")
        lines.append(f"**개발 프로그램**: {program}")

        if is_tod:
            lines.append(f"**역세권 개발**: {subway_name} 역 {subway_dist:.0f}m 이내 - 고밀 복합개발 적용")

        # Reference cases
        if similar_cases:
            best = similar_cases[0]
            lines.append(f"\n**참조 사례**: {best['name']} (유사도 {best['similarity_score']:.0%})")
            lines.append(f"- 시사점: {best.get('lessons', '')}")

        # Zoning breakdown
        lines.append("\n**용도지역 배치 원칙**:")
        lines.append("- 접면도로 가장 넓은 블록 → 상업지역 배치")
        lines.append("- 역세권 내 블록 → 준주거/제3종일반주거 고밀 배치")
        lines.append("- 북측 블록 → 제1-2종일반주거 (일조권 보호)")
        lines.append("- 상업-주거 완충 → 근린상업/준주거 배치")

        # Stats summary
        lines.append("\n**면적 배분 결과**:")
        for code, stat in zoning_stats.items():
            lines.append(f"- {stat['name']}: {stat['area_ha']:.2f}ha ({stat['ratio']*100:.1f}%)")

        # Context
        adj_n = context_analysis.get('adjacent_use_north', '미확인')
        adj_s = context_analysis.get('adjacent_use_south', '미확인')
        lines.append(f"\n**주변 맥락 고려**:")
        lines.append(f"- 북측: {adj_n} | 남측: {adj_s}")

        return '\n'.join(lines)
