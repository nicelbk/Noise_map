"""
도로망 생성 모듈
"""

import json
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import (
    shape, Polygon, LineString, MultiLineString, MultiPolygon, Point
)
from shapely.ops import unary_union, split, polygonize
from pyproj import Transformer

EPSG_WGS84 = 4326
EPSG_5179 = 5179

transformer_to_5179 = Transformer.from_crs(EPSG_WGS84, EPSG_5179, always_xy=True)
transformer_to_wgs84 = Transformer.from_crs(EPSG_5179, EPSG_WGS84, always_xy=True)


def _empty_gdf(columns=None) -> gpd.GeoDataFrame:
    cols = columns or []
    data = {c: [] for c in cols}
    data['geometry'] = []
    return gpd.GeoDataFrame(data, crs=f"EPSG:{EPSG_5179}")


class RoadGenerator:

    def generate(
        self,
        boundary_geojson: dict,
        entry_points: list,
        target_road_ratio: float = 0.25
    ) -> dict:
        """
        반환:
        {
          'road_lines_gdf': GeoDataFrame,   # 도로 중심선
          'road_polygons_gdf': GeoDataFrame, # 도로 면적(폭 적용)
          'road_geojson': dict,
          'road_stats': {
            'total_length_m': float,
            'road_area_m2': float,
            'road_ratio': float,
            'by_hierarchy': dict
          }
        }
        """
        try:
            geom = shape(boundary_geojson)
            coords = list(geom.exterior.coords)
            transformed = [transformer_to_5179.transform(lon, lat) for lon, lat in coords]
            boundary_geom = Polygon(transformed)
            boundary_area = boundary_geom.area

            # 1. Generate primary roads
            primary_gdf = self._generate_primary_roads(boundary_geom, entry_points)

            # 2. Generate secondary roads
            secondary_gdf = self._generate_secondary_roads(
                boundary_geom, primary_gdf, target_block_area_m2=10000
            )

            # 3. Combine and generate local roads
            all_roads = self._concat_gdfs([primary_gdf, secondary_gdf])
            local_gdf = self._generate_local_roads(
                boundary_geom, all_roads, target_block_area_m2=5000
            )

            # 4. Combine all roads
            road_lines_gdf = self._concat_gdfs([primary_gdf, secondary_gdf, local_gdf])

            # 5. Apply road widths → polygons
            road_polys = []
            for idx, row in road_lines_gdf.iterrows():
                poly = self._apply_road_width(row.geometry, row['width_m'])
                if poly is not None and not poly.is_empty:
                    road_polys.append({
                        'geometry': poly.intersection(boundary_geom),
                        'width_m': row['width_m'],
                        'hierarchy': row['hierarchy'],
                    })

            if road_polys:
                road_polygons_gdf = gpd.GeoDataFrame(road_polys, crs=f"EPSG:{EPSG_5179}")
                # Remove empty geometries
                road_polygons_gdf = road_polygons_gdf[
                    ~road_polygons_gdf.geometry.is_empty
                ].reset_index(drop=True)
            else:
                road_polygons_gdf = _empty_gdf(['width_m', 'hierarchy'])

            # 6. Check and adjust road ratio
            current_ratio = self._check_road_ratio(road_polygons_gdf, boundary_area)
            if current_ratio < 0.18 or current_ratio > 0.32:
                road_lines_gdf, road_polygons_gdf = self._adjust_road_ratio(
                    road_lines_gdf, road_polygons_gdf, boundary_area, target_road_ratio
                )

            # 7. Calculate statistics
            road_stats = self._calculate_stats(
                road_lines_gdf, road_polygons_gdf, boundary_area
            )

            # 8. Convert to WGS84 GeoJSON
            road_geojson = self._to_geojson(road_polygons_gdf)

            return {
                'road_lines_gdf': road_lines_gdf,
                'road_polygons_gdf': road_polygons_gdf,
                'road_geojson': road_geojson,
                'road_stats': road_stats,
            }

        except Exception as e:
            empty_lines = _empty_gdf(['width_m', 'hierarchy'])
            empty_polys = _empty_gdf(['width_m', 'hierarchy'])
            return {
                'road_lines_gdf': empty_lines,
                'road_polygons_gdf': empty_polys,
                'road_geojson': {'type': 'FeatureCollection', 'features': []},
                'road_stats': {
                    'total_length_m': 0,
                    'road_area_m2': 0,
                    'road_ratio': 0,
                    'by_hierarchy': {},
                },
            }

    def _generate_primary_roads(
        self,
        boundary_geom: Polygon,
        entry_points: list
    ) -> gpd.GeoDataFrame:
        """
        주간선도로 생성
        entry_points를 연결하는 주축 1~2개
        폭원: 20~35m
        """
        minx, miny, maxx, maxy = boundary_geom.bounds
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2
        width = maxx - minx
        height = maxy - miny

        roads = []

        if entry_points and len(entry_points) >= 2:
            # Connect main entry points
            # Sort by road width (widest first)
            sorted_entries = sorted(
                entry_points, key=lambda e: e.get('road_width_m', 0), reverse=True
            )
            main_entries = sorted_entries[:2]

            pt1 = Point(main_entries[0]['point'][0], main_entries[0]['point'][1])
            pt2 = Point(main_entries[1]['point'][0], main_entries[1]['point'][1])

            # Primary road connecting entry points through center
            mid_x = (pt1.x + pt2.x) / 2
            mid_y = (pt1.y + pt2.y) / 2

            # Ensure line passes through boundary center area
            line = LineString([
                (pt1.x, pt1.y),
                (cx, cy),
                (pt2.x, pt2.y)
            ])
            clipped = line.intersection(boundary_geom)
            if not clipped.is_empty and clipped.length > 50:
                if clipped.geom_type == 'LineString':
                    roads.append({
                        'geometry': clipped,
                        'width_m': 25.0,
                        'hierarchy': 'primary',
                    })
                elif clipped.geom_type == 'MultiLineString':
                    for seg in clipped.geoms:
                        if seg.length > 50:
                            roads.append({
                                'geometry': seg,
                                'width_m': 25.0,
                                'hierarchy': 'primary',
                            })

        # Add main cross axis if no entry points or only 1
        if len(roads) == 0 or len(entry_points) <= 1:
            # Horizontal main road through center
            h_road = LineString([(minx, cy), (maxx, cy)])
            clipped_h = h_road.intersection(boundary_geom)
            if not clipped_h.is_empty and clipped_h.length > 50:
                roads.append({
                    'geometry': clipped_h,
                    'width_m': 25.0,
                    'hierarchy': 'primary',
                })

            # Vertical main road through center (if wide enough)
            if width > height * 0.6:
                v_road = LineString([(cx, miny), (cx, maxy)])
                clipped_v = v_road.intersection(boundary_geom)
                if not clipped_v.is_empty and clipped_v.length > 50:
                    roads.append({
                        'geometry': clipped_v,
                        'width_m': 20.0,
                        'hierarchy': 'primary',
                    })

        if not roads:
            return _empty_gdf(['width_m', 'hierarchy'])

        return gpd.GeoDataFrame(roads, crs=f"EPSG:{EPSG_5179}")

    def _generate_secondary_roads(
        self,
        boundary_geom: Polygon,
        primary_roads_gdf: gpd.GeoDataFrame,
        target_block_area_m2: float = 10000
    ) -> gpd.GeoDataFrame:
        """
        보조간선도로 생성
        주간선 기준 블록 면적이 target_block_area_m2 될 때까지 분할
        폭원: 12~20m
        """
        minx, miny, maxx, maxy = boundary_geom.bounds
        width = maxx - minx
        height = maxy - miny

        roads = []

        # Determine grid spacing based on target block area
        # block_area ≈ spacing_x * spacing_y → use sqrt for square blocks
        target_block_side = math.sqrt(target_block_area_m2)

        # Horizontal secondary roads
        n_h = max(1, int(height / target_block_side) - 1)
        for i in range(1, n_h + 1):
            y = miny + (height * i) / (n_h + 1)
            line = LineString([(minx, y), (maxx, y)])
            clipped = line.intersection(boundary_geom)
            if not clipped.is_empty and clipped.length > 30:
                geoms = [clipped] if clipped.geom_type == 'LineString' else list(clipped.geoms)
                for g in geoms:
                    if g.length > 30:
                        roads.append({
                            'geometry': g,
                            'width_m': 15.0,
                            'hierarchy': 'secondary',
                        })

        # Vertical secondary roads
        n_v = max(1, int(width / target_block_side) - 1)
        for i in range(1, n_v + 1):
            x = minx + (width * i) / (n_v + 1)
            line = LineString([(x, miny), (x, maxy)])
            clipped = line.intersection(boundary_geom)
            if not clipped.is_empty and clipped.length > 30:
                geoms = [clipped] if clipped.geom_type == 'LineString' else list(clipped.geoms)
                for g in geoms:
                    if g.length > 30:
                        roads.append({
                            'geometry': g,
                            'width_m': 15.0,
                            'hierarchy': 'secondary',
                        })

        if not roads:
            return _empty_gdf(['width_m', 'hierarchy'])

        return gpd.GeoDataFrame(roads, crs=f"EPSG:{EPSG_5179}")

    def _generate_local_roads(
        self,
        boundary_geom: Polygon,
        all_roads_gdf: gpd.GeoDataFrame,
        target_block_area_m2: float = 5000
    ) -> gpd.GeoDataFrame:
        """
        국지도로 생성
        블록 면적이 target_block_area_m2 초과하는 경우 추가 분할
        막힌도로 35m 이하 제한
        폭원: 6~12m
        """
        minx, miny, maxx, maxy = boundary_geom.bounds
        width = maxx - minx
        height = maxy - miny

        # Use finer grid for local roads
        target_block_side = math.sqrt(target_block_area_m2)

        # Existing road lines as union for block detection
        existing_lines = []
        if not all_roads_gdf.empty:
            for geom in all_roads_gdf.geometry:
                existing_lines.append(geom)

        roads = []

        # Horizontal local roads (between secondary roads)
        n_h = max(2, int(height / target_block_side))
        for i in range(1, n_h):
            y = miny + (height * i) / n_h
            line = LineString([(minx, y), (maxx, y)])
            clipped = line.intersection(boundary_geom)

            # Check if line is too close to existing road
            too_close = False
            for existing in existing_lines:
                if clipped.distance(existing) < 30:
                    too_close = True
                    break

            if not too_close and not clipped.is_empty and clipped.length > 20:
                geoms = [clipped] if clipped.geom_type == 'LineString' else list(clipped.geoms)
                for g in geoms:
                    # Enforce dead-end limit (35m max)
                    if g.length <= 35 or g.length > 100:  # skip very short, keep reasonable
                        if g.length > 20:
                            roads.append({
                                'geometry': g,
                                'width_m': 8.0,
                                'hierarchy': 'local',
                            })

        # Vertical local roads
        n_v = max(2, int(width / target_block_side))
        for i in range(1, n_v):
            x = minx + (width * i) / n_v
            line = LineString([(x, miny), (x, maxy)])
            clipped = line.intersection(boundary_geom)

            too_close = False
            for existing in existing_lines:
                if clipped.distance(existing) < 30:
                    too_close = True
                    break

            if not too_close and not clipped.is_empty and clipped.length > 20:
                geoms = [clipped] if clipped.geom_type == 'LineString' else list(clipped.geoms)
                for g in geoms:
                    if g.length > 20:
                        roads.append({
                            'geometry': g,
                            'width_m': 8.0,
                            'hierarchy': 'local',
                        })

        if not roads:
            return _empty_gdf(['width_m', 'hierarchy'])

        return gpd.GeoDataFrame(roads, crs=f"EPSG:{EPSG_5179}")

    def _apply_road_width(
        self,
        road_line,
        width_m: float
    ):
        """
        도로 중심선에 폭원 적용 → Polygon 반환
        shapely buffer(width_m/2) 사용
        """
        try:
            if road_line is None or road_line.is_empty:
                return None
            buffered = road_line.buffer(width_m / 2, cap_style=2, join_style=2)
            return buffered
        except Exception:
            return None

    def _check_road_ratio(
        self,
        road_polygons_gdf: gpd.GeoDataFrame,
        boundary_area_m2: float
    ) -> float:
        """
        현재 도로율 계산 후 반환
        """
        if road_polygons_gdf.empty or boundary_area_m2 <= 0:
            return 0.0
        try:
            road_union = unary_union(road_polygons_gdf.geometry.values)
            road_area = road_union.area
            return road_area / boundary_area_m2
        except Exception:
            return 0.0

    def _adjust_road_ratio(
        self,
        road_lines_gdf: gpd.GeoDataFrame,
        road_polygons_gdf: gpd.GeoDataFrame,
        boundary_area_m2: float,
        target_ratio: float = 0.25
    ) -> tuple:
        """
        도로율 조정
        초과(>30%): 국지도로부터 순서대로 제거
        미달(<20%): 가장 큰 블록 추가 분할
        반환: (조정된 road_lines_gdf, road_polygons_gdf)
        """
        try:
            current = self._check_road_ratio(road_polygons_gdf, boundary_area_m2)

            if current > 0.30:
                # Remove local roads one by one until ratio drops
                local_mask = road_lines_gdf['hierarchy'] == 'local'
                local_indices = road_lines_gdf[local_mask].index.tolist()

                for idx in local_indices:
                    if current <= 0.28:
                        break
                    road_lines_gdf = road_lines_gdf.drop(idx)
                    # Rebuild polygons
                    road_polys = []
                    for ridx, row in road_lines_gdf.iterrows():
                        poly = self._apply_road_width(row.geometry, row['width_m'])
                        if poly is not None and not poly.is_empty:
                            road_polys.append({
                                'geometry': poly,
                                'width_m': row['width_m'],
                                'hierarchy': row['hierarchy'],
                            })
                    if road_polys:
                        road_polygons_gdf = gpd.GeoDataFrame(road_polys, crs=f"EPSG:{EPSG_5179}")
                    current = self._check_road_ratio(road_polygons_gdf, boundary_area_m2)

            return road_lines_gdf, road_polygons_gdf
        except Exception:
            return road_lines_gdf, road_polygons_gdf

    def _calculate_stats(
        self,
        road_lines_gdf: gpd.GeoDataFrame,
        road_polygons_gdf: gpd.GeoDataFrame,
        boundary_area_m2: float
    ) -> dict:
        """도로 통계 계산"""
        total_length = 0.0
        by_hierarchy = {}

        if not road_lines_gdf.empty:
            total_length = float(road_lines_gdf.geometry.length.sum())
            for hier in road_lines_gdf['hierarchy'].unique():
                mask = road_lines_gdf['hierarchy'] == hier
                by_hierarchy[hier] = {
                    'count': int(mask.sum()),
                    'total_length_m': round(float(road_lines_gdf[mask].geometry.length.sum()), 1),
                }

        road_area = 0.0
        if not road_polygons_gdf.empty:
            try:
                road_union = unary_union(road_polygons_gdf.geometry.values)
                road_area = float(road_union.area)
            except Exception:
                road_area = float(road_polygons_gdf.geometry.area.sum())

        road_ratio = road_area / boundary_area_m2 if boundary_area_m2 > 0 else 0.0

        return {
            'total_length_m': round(total_length, 1),
            'road_area_m2': round(road_area, 1),
            'road_ratio': round(road_ratio, 4),
            'by_hierarchy': by_hierarchy,
        }

    def _to_geojson(self, road_polygons_gdf: gpd.GeoDataFrame) -> dict:
        """WGS84 GeoJSON 변환"""
        try:
            if road_polygons_gdf.empty:
                return {'type': 'FeatureCollection', 'features': []}
            wgs84 = road_polygons_gdf.to_crs(EPSG_WGS84)
            return json.loads(wgs84.to_json())
        except Exception:
            return {'type': 'FeatureCollection', 'features': []}

    def _concat_gdfs(self, gdfs: list) -> gpd.GeoDataFrame:
        """여러 GeoDataFrame 합치기"""
        non_empty = [g for g in gdfs if not g.empty]
        if not non_empty:
            return _empty_gdf(['width_m', 'hierarchy'])
        import pandas as pd
        return gpd.GeoDataFrame(
            pd.concat(non_empty, ignore_index=True),
            crs=f"EPSG:{EPSG_5179}"
        )
