"""
도시계획 법규 검증 모듈
국토계획법 및 서울시 도시계획 조례 기준
"""

import geopandas as gpd
from shapely.geometry import shape, Polygon
from pyproj import Transformer

EPSG_WGS84 = 4326
EPSG_5179 = 5179

transformer_to_5179 = Transformer.from_crs(EPSG_WGS84, EPSG_5179, always_xy=True)


class PlanValidator:

    def validate(
        self,
        boundary_geojson: dict,
        road_stats: dict,
        blocks_gdf: gpd.GeoDataFrame,
        facilities_gdf: gpd.GeoDataFrame,
        zoning_stats: dict
    ) -> dict:
        """
        반환:
        {
          'is_valid': bool,
          'score': int,           # 0~100점
          'required': [
            {'item': str, 'pass': bool, 'value': str, 'standard': str}
          ],
          'recommended': [
            {'item': str, 'pass': bool, 'value': str, 'standard': str}
          ],
          'issues': list[dict]
        }
        """
        try:
            geom = shape(boundary_geojson)
            coords = list(geom.exterior.coords)
            transformed = [transformer_to_5179.transform(lon, lat) for lon, lat in coords]
            boundary_geom = Polygon(transformed)
            total_area_m2 = boundary_geom.area
            total_area_ha = total_area_m2 / 10000

            required_checks = []
            recommended_checks = []
            issues = []

            # ==================
            # REQUIRED CHECKS
            # ==================

            # 1. 도로율 (20~30%)
            road_ratio = road_stats.get('road_ratio', 0)
            road_pct = road_ratio * 100
            road_pass = 0.20 <= road_ratio <= 0.30
            required_checks.append({
                'item': '도로율',
                'pass': road_pass,
                'value': f'{road_pct:.1f}%',
                'standard': '20~30%',
            })
            if not road_pass:
                issues.append({
                    'severity': 'error',
                    'item': '도로율',
                    'message': f'도로율 {road_pct:.1f}%는 기준 범위(20~30%)를 벗어남',
                })

            # 2. 녹지율 (최소 10%)
            green_area = sum(
                s.get('area_m2', 0)
                for code, s in zoning_stats.items()
                if code in ('GN', 'PK')
            )
            facility_park_area = 0.0
            if not facilities_gdf.empty and 'facility_type' in facilities_gdf.columns:
                park_types = ['neighborhood_park', 'children_park']
                park_gdf = facilities_gdf[facilities_gdf['facility_type'].isin(park_types)]
                facility_park_area = float(park_gdf.geometry.area.sum())

            total_green = green_area + facility_park_area
            green_ratio = total_green / total_area_m2 if total_area_m2 > 0 else 0
            green_pct = green_ratio * 100
            green_pass = green_ratio >= 0.10
            required_checks.append({
                'item': '녹지율',
                'pass': green_pass,
                'value': f'{green_pct:.1f}%',
                'standard': '최소 10%',
            })
            if not green_pass:
                issues.append({
                    'severity': 'error',
                    'item': '녹지율',
                    'message': f'녹지율 {green_pct:.1f}%가 최소 기준(10%) 미달',
                })

            # 3. 블록 수 (최소 2개 이상)
            block_count = len(blocks_gdf) if not blocks_gdf.empty else 0
            block_pass = block_count >= 2
            required_checks.append({
                'item': '블록 분할',
                'pass': block_pass,
                'value': f'{block_count}개',
                'standard': '최소 2개',
            })

            # 4. 최소 진입도로 (접면도로 폭 6m 이상)
            if not blocks_gdf.empty and 'adjacent_road_width_m' in blocks_gdf.columns:
                min_road_width = float(blocks_gdf['adjacent_road_width_m'].min())
                access_pass = min_road_width >= 6.0 or block_count <= 1
            else:
                min_road_width = 0.0
                access_pass = False
            required_checks.append({
                'item': '최소 접면도로',
                'pass': access_pass,
                'value': f'{min_road_width:.1f}m',
                'standard': '6m 이상',
            })
            if not access_pass:
                issues.append({
                    'severity': 'error',
                    'item': '접면도로',
                    'message': f'일부 블록 접면도로 폭({min_road_width:.1f}m)이 6m 미달',
                })

            # 5. 공원 면적 (주거인구 1인당 3㎡ 이상)
            total_green_for_park = total_green
            # Estimate residential population
            res_area = sum(
                s.get('area_m2', 0)
                for code, s in zoning_stats.items()
                if code in ('R1', 'R2', 'R3', 'R4', 'R5', 'RQ')
            )
            # Estimate population from residential area
            est_pop = max(100, int(res_area * 0.015))  # ~150 persons/ha → 0.015/m²
            park_per_person = total_green_for_park / est_pop if est_pop > 0 else 0
            park_std_pass = park_per_person >= 3.0
            required_checks.append({
                'item': '1인당 공원면적',
                'pass': park_std_pass,
                'value': f'{park_per_person:.1f}㎡/인',
                'standard': '3㎡/인 이상',
            })
            if not park_std_pass:
                issues.append({
                    'severity': 'warning',
                    'item': '공원면적',
                    'message': f'1인당 공원면적 {park_per_person:.1f}㎡가 권장 기준(3㎡) 미달',
                })

            # ==================
            # RECOMMENDED CHECKS
            # ==================

            # 6. 주간선도로 폭원 (20m 이상)
            by_hierarchy = road_stats.get('by_hierarchy', {})
            has_primary = 'primary' in by_hierarchy
            primary_pass = has_primary
            recommended_checks.append({
                'item': '주간선도로',
                'pass': primary_pass,
                'value': '있음' if has_primary else '없음',
                'standard': '폭원 20m 이상 1개 이상',
            })

            # 7. 공원 서비스 반경 (250m 커버리지 70% 이상)
            park_coverage_250 = 0.0
            if not facilities_gdf.empty:
                # Already calculated in facility placer - use rough estimate here
                park_types = ['neighborhood_park', 'children_park']
                if 'facility_type' in facilities_gdf.columns:
                    parks = facilities_gdf[facilities_gdf['facility_type'].isin(park_types)]
                    if not parks.empty and not blocks_gdf.empty:
                        total_block_area = float(blocks_gdf.geometry.area.sum())
                        park_zones = parks.geometry.buffer(250)
                        from shapely.ops import unary_union
                        union_zone = unary_union(park_zones)
                        covered = float(
                            blocks_gdf.geometry.intersection(union_zone).area.sum()
                        )
                        park_coverage_250 = covered / total_block_area if total_block_area > 0 else 0
            park_cov_pass = park_coverage_250 >= 0.70
            recommended_checks.append({
                'item': '공원 서비스 반경(250m)',
                'pass': park_cov_pass,
                'value': f'{park_coverage_250*100:.1f}%',
                'standard': '70% 이상',
            })
            if not park_cov_pass:
                issues.append({
                    'severity': 'info',
                    'item': '공원 접근성',
                    'message': f'공원 250m 커버리지 {park_coverage_250*100:.1f}%가 권장 기준(70%) 미달',
                })

            # 8. 블록 평균 크기 (3,000~15,000㎡)
            avg_block_area = 0.0
            if not blocks_gdf.empty and 'area_m2' in blocks_gdf.columns:
                avg_block_area = float(blocks_gdf['area_m2'].mean())
            block_size_pass = 3000 <= avg_block_area <= 15000
            recommended_checks.append({
                'item': '평균 블록 크기',
                'pass': block_size_pass,
                'value': f'{avg_block_area:.0f}㎡',
                'standard': '3,000~15,000㎡',
            })

            # 9. 상업지역 비율 (30% 이하)
            commercial_area = sum(
                s.get('area_m2', 0)
                for code, s in zoning_stats.items()
                if code in ('CB', 'CG', 'CN')
            )
            commercial_ratio = commercial_area / total_area_m2 if total_area_m2 > 0 else 0
            commercial_pass = commercial_ratio <= 0.30
            recommended_checks.append({
                'item': '상업지역 비율',
                'pass': commercial_pass,
                'value': f'{commercial_ratio*100:.1f}%',
                'standard': '30% 이하',
            })

            # 10. 도로망 연결성 (보조간선도로 존재)
            has_secondary = 'secondary' in by_hierarchy
            conn_pass = has_secondary
            recommended_checks.append({
                'item': '도로망 위계',
                'pass': conn_pass,
                'value': '있음' if has_secondary else '없음',
                'standard': '보조간선도로 포함',
            })

            # ==================
            # SCORE CALCULATION
            # ==================
            required_passed = sum(1 for c in required_checks if c['pass'])
            recommended_passed = sum(1 for c in recommended_checks if c['pass'])

            required_score = (required_passed / len(required_checks)) * 60 if required_checks else 0
            recommended_score = (recommended_passed / len(recommended_checks)) * 40 if recommended_checks else 0
            total_score = int(required_score + recommended_score)

            is_valid = required_passed >= len(required_checks) - 1  # 필수 1개 이하 실패

            return {
                'is_valid': is_valid,
                'score': total_score,
                'required': required_checks,
                'recommended': recommended_checks,
                'issues': issues,
            }

        except Exception as e:
            return {
                'is_valid': False,
                'score': 0,
                'required': [],
                'recommended': [],
                'issues': [{'severity': 'error', 'item': '검증 오류', 'message': str(e)}],
            }
