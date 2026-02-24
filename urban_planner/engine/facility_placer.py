"""
공공시설 배치 모듈
절대 규칙: Circle/Point 사용 금지, 모든 시설은 블록 Polygon 그대로 사용
"""

import json
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from pyproj import Transformer

EPSG_WGS84 = 4326
EPSG_5179 = 5179

transformer_to_wgs84 = Transformer.from_crs(EPSG_5179, EPSG_WGS84, always_xy=True)

FACILITY_COLORS = {
    'neighborhood_park': '#2ECC71',    # 근린공원 - 초록
    'children_park': '#27AE60',        # 어린이공원 - 진초록
    'elementary_school': '#3498DB',    # 초등학교 - 파랑
    'middle_school': '#2980B9',        # 중학교 - 진파랑
    'high_school': '#1ABC9C',          # 고등학교 - 청록
    'community_center': '#E74C3C',     # 주민센터 - 빨강
    'welfare_center': '#E67E22',       # 복지관 - 주황
    'park': '#2ECC71',
}

FACILITY_NAMES = {
    'neighborhood_park': '근린공원',
    'children_park': '어린이공원',
    'elementary_school': '초등학교',
    'middle_school': '중학교',
    'high_school': '고등학교',
    'community_center': '주민센터',
    'welfare_center': '복지관',
}


class FacilityPlacer:

    def place(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        design_params: dict,
        context: dict
    ) -> dict:
        """
        공공시설 배치

        반환:
        {
          'facilities_gdf': GeoDataFrame,
          'facilities_geojson': dict,
          'coverage': {
            'park_250m_ratio': float,
            'park_500m_ratio': float,
            'school_500m_ratio': float
          }
        }

        절대 규칙:
        - Circle/Point 사용 금지
        - 모든 시설은 블록 Polygon 그대로 사용
        - 블록에 facility_type 속성 추가
        """
        try:
            if blocks_gdf.empty:
                return self._empty_result()

            target_population = design_params.get('target_population', 0)
            area_ha = blocks_gdf.geometry.area.sum() / 10000

            # Estimate population if not given
            if target_population <= 0:
                # Rough estimate: 150 persons/ha for mixed development
                target_population = int(area_ha * 150)

            # Estimate households (avg 2.5 persons/household)
            estimated_households = max(100, int(target_population / 2.5))

            facility_rows = []

            # 1. Select park blocks
            nbhd_parks = self._select_park_blocks(blocks_gdf, 'neighborhood')
            for idx, row in nbhd_parks.iterrows():
                facility_rows.append({
                    'geometry': row.geometry,
                    'facility_type': 'neighborhood_park',
                    'facility_name': '근린공원',
                    'area_m2': row.geometry.area,
                    'color': FACILITY_COLORS['neighborhood_park'],
                })

            child_parks = self._select_park_blocks(blocks_gdf, 'children')
            for idx, row in child_parks.iterrows():
                facility_rows.append({
                    'geometry': row.geometry,
                    'facility_type': 'children_park',
                    'facility_name': '어린이공원',
                    'area_m2': row.geometry.area,
                    'color': FACILITY_COLORS['children_park'],
                })

            # 2. Select school blocks
            school_blocks = self._select_school_blocks(blocks_gdf, estimated_households)
            for idx, row in school_blocks.iterrows():
                school_type = row.get('_school_type', 'elementary_school')
                facility_rows.append({
                    'geometry': row.geometry,
                    'facility_type': school_type,
                    'facility_name': FACILITY_NAMES.get(school_type, '학교'),
                    'area_m2': row.geometry.area,
                    'color': FACILITY_COLORS.get(school_type, '#3498DB'),
                })

            # 3. Community facilities (주민센터, 복지관)
            if estimated_households >= 500:
                community_blocks = self._select_community_blocks(blocks_gdf, estimated_households)
                for idx, row in community_blocks.iterrows():
                    ftype = row.get('_facility_type', 'community_center')
                    facility_rows.append({
                        'geometry': row.geometry,
                        'facility_type': ftype,
                        'facility_name': FACILITY_NAMES.get(ftype, '공공시설'),
                        'area_m2': row.geometry.area,
                        'color': FACILITY_COLORS.get(ftype, '#E74C3C'),
                    })

            if not facility_rows:
                return self._empty_result()

            facilities_gdf = gpd.GeoDataFrame(facility_rows, crs=f"EPSG:{EPSG_5179}")

            # Calculate coverage
            coverage = self._calculate_coverage(
                blocks_gdf, facilities_gdf, target_population
            )

            # Convert to WGS84 GeoJSON
            facilities_wgs84 = facilities_gdf.to_crs(EPSG_WGS84)
            facilities_geojson = json.loads(facilities_wgs84.to_json())

            return {
                'facilities_gdf': facilities_gdf,
                'facilities_geojson': facilities_geojson,
                'coverage': coverage,
            }

        except Exception as e:
            return self._empty_result()

    def _select_park_blocks(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        park_type: str  # 'neighborhood'|'children'
    ) -> gpd.GeoDataFrame:
        """
        공원으로 지정할 블록 선택
        neighborhood: 10,000m² 이상 블록 또는 인접블록 합치기
        children: 1,500~5,000m² 주거인접 블록
        """
        try:
            if park_type == 'neighborhood':
                # Large blocks for neighborhood park
                candidates = blocks_gdf[blocks_gdf['area_m2'] >= 8000].copy()
                if candidates.empty:
                    # Use largest block
                    largest = blocks_gdf.nlargest(1, 'area_m2')
                    return largest
                # Select 1-2 blocks
                selected = candidates.nlargest(min(2, len(candidates)), 'area_m2')
                return selected

            else:  # children
                # Small to medium blocks for children's park
                candidates = blocks_gdf[
                    (blocks_gdf['area_m2'] >= 1500) &
                    (blocks_gdf['area_m2'] <= 6000)
                ].copy()

                if candidates.empty:
                    return gpd.GeoDataFrame(crs=f"EPSG:{EPSG_5179}")

                # Prefer blocks near residential areas
                # Select up to 2 blocks spread out spatially
                if len(candidates) <= 2:
                    return candidates

                # Select blocks that are spatially separated
                selected = []
                remaining = candidates.copy()
                while len(selected) < 3 and not remaining.empty:
                    # Pick the one with smallest area (appropriate for children's park)
                    pick = remaining.nsmallest(1, 'area_m2').iloc[0]
                    pick_geom = pick.geometry
                    selected.append(pick.name)

                    # Remove nearby blocks (within 200m)
                    centroid = pick_geom.centroid
                    remaining = remaining[
                        remaining.geometry.centroid.distance(centroid) > 200
                    ]

                return candidates.loc[selected]

        except Exception:
            return gpd.GeoDataFrame(crs=f"EPSG:{EPSG_5179}")

    def _select_school_blocks(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        estimated_households: int
    ) -> gpd.GeoDataFrame:
        """
        학교 블록 선택
        초등: 500세대당 1개, 5,000~10,000m²
        중학: 2,000세대당 1개, 8,000~15,000m²
        """
        try:
            selected_rows = []

            # Elementary schools
            n_elementary = max(1, estimated_households // 500)
            n_elementary = min(n_elementary, 3)  # max 3

            elem_candidates = blocks_gdf[
                (blocks_gdf['area_m2'] >= 4000) &
                (blocks_gdf['area_m2'] <= 12000)
            ].copy()

            if not elem_candidates.empty:
                elem_selected = elem_candidates.nlargest(
                    min(n_elementary, len(elem_candidates)), 'area_m2'
                )
                for idx, row in elem_selected.iterrows():
                    row_dict = row.to_dict()
                    row_dict['_school_type'] = 'elementary_school'
                    selected_rows.append(row_dict)

            # Middle schools
            n_middle = max(0, estimated_households // 2000)
            n_middle = min(n_middle, 2)  # max 2

            if n_middle > 0:
                # Exclude already selected
                selected_indices = [r.get('_orig_idx') for r in selected_rows]
                middle_candidates = blocks_gdf[
                    (blocks_gdf['area_m2'] >= 8000) &
                    (blocks_gdf['area_m2'] <= 15000)
                ].copy()

                if not middle_candidates.empty:
                    # Select blocks not already used
                    available = middle_candidates[
                        ~middle_candidates.index.isin([
                            r for r in selected_rows
                        ])
                    ]
                    if available.empty:
                        available = middle_candidates

                    mid_selected = available.nlargest(
                        min(n_middle, len(available)), 'area_m2'
                    )
                    for idx, row in mid_selected.iterrows():
                        row_dict = row.to_dict()
                        row_dict['_school_type'] = 'middle_school'
                        selected_rows.append(row_dict)

            if not selected_rows:
                return gpd.GeoDataFrame(crs=f"EPSG:{EPSG_5179}")

            result_gdf = gpd.GeoDataFrame(selected_rows, crs=f"EPSG:{EPSG_5179}")
            return result_gdf

        except Exception:
            return gpd.GeoDataFrame(crs=f"EPSG:{EPSG_5179}")

    def _select_community_blocks(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        estimated_households: int
    ) -> gpd.GeoDataFrame:
        """주민센터, 복지관 블록 선택"""
        try:
            selected_rows = []

            # 주민센터: 2,000세대당 1개
            n_community = max(1, estimated_households // 2000)
            n_community = min(n_community, 2)

            candidates = blocks_gdf[
                (blocks_gdf['area_m2'] >= 1000) &
                (blocks_gdf['area_m2'] <= 3000)
            ].copy()

            if candidates.empty:
                candidates = blocks_gdf[
                    blocks_gdf['area_m2'] < 5000
                ].nsmallest(3, 'area_m2')

            if not candidates.empty:
                selected = candidates.nsmallest(
                    min(n_community, len(candidates)), 'area_m2'
                )
                for idx, row in selected.iterrows():
                    row_dict = row.to_dict()
                    row_dict['_facility_type'] = 'community_center'
                    selected_rows.append(row_dict)

            # 복지관: 10,000명당 1개
            if estimated_households * 2.5 >= 10000:
                welfare_candidates = blocks_gdf[
                    (blocks_gdf['area_m2'] >= 2000) &
                    (blocks_gdf['area_m2'] <= 5000)
                ].copy()

                if not welfare_candidates.empty:
                    welfare = welfare_candidates.nsmallest(1, 'area_m2')
                    for idx, row in welfare.iterrows():
                        row_dict = row.to_dict()
                        row_dict['_facility_type'] = 'welfare_center'
                        selected_rows.append(row_dict)

            if not selected_rows:
                return gpd.GeoDataFrame(crs=f"EPSG:{EPSG_5179}")

            return gpd.GeoDataFrame(selected_rows, crs=f"EPSG:{EPSG_5179}")

        except Exception:
            return gpd.GeoDataFrame(crs=f"EPSG:{EPSG_5179}")

    def _calculate_coverage(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        facilities_gdf: gpd.GeoDataFrame,
        target_population: int
    ) -> dict:
        """서비스 커버리지 계산"""
        try:
            if facilities_gdf.empty or blocks_gdf.empty:
                return {
                    'park_250m_ratio': 0.0,
                    'park_500m_ratio': 0.0,
                    'school_500m_ratio': 0.0,
                }

            total_area = blocks_gdf.geometry.area.sum()

            park_types = ['neighborhood_park', 'children_park']
            park_mask = facilities_gdf['facility_type'].isin(park_types)
            parks_gdf = facilities_gdf[park_mask]

            school_types = ['elementary_school', 'middle_school']
            school_mask = facilities_gdf['facility_type'].isin(school_types)
            schools_gdf = facilities_gdf[school_mask]

            def calc_coverage(service_gdf, radius_m):
                if service_gdf.empty:
                    return 0.0
                service_zones = service_gdf.geometry.buffer(radius_m)
                union_zone = unary_union(service_zones)
                covered = blocks_gdf.geometry.intersection(union_zone).area.sum()
                return round(float(covered) / float(total_area), 4) if total_area > 0 else 0.0

            park_250 = calc_coverage(parks_gdf, 250)
            park_500 = calc_coverage(parks_gdf, 500)
            school_500 = calc_coverage(schools_gdf, 500)

            return {
                'park_250m_ratio': park_250,
                'park_500m_ratio': park_500,
                'school_500m_ratio': school_500,
            }
        except Exception:
            return {
                'park_250m_ratio': 0.0,
                'park_500m_ratio': 0.0,
                'school_500m_ratio': 0.0,
            }

    def _empty_result(self) -> dict:
        empty_gdf = gpd.GeoDataFrame(
            columns=['geometry', 'facility_type', 'facility_name', 'area_m2', 'color'],
            crs=f"EPSG:{EPSG_5179}"
        )
        return {
            'facilities_gdf': empty_gdf,
            'facilities_geojson': {'type': 'FeatureCollection', 'features': []},
            'coverage': {
                'park_250m_ratio': 0.0,
                'park_500m_ratio': 0.0,
                'school_500m_ratio': 0.0,
            },
        }
