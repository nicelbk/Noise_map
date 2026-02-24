"""
도로망으로 대상지를 블록으로 분할하는 모듈
"""

import json
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize, split
from pyproj import Transformer

EPSG_WGS84 = 4326
EPSG_5179 = 5179

transformer_to_5179 = Transformer.from_crs(EPSG_WGS84, EPSG_5179, always_xy=True)
transformer_to_wgs84 = Transformer.from_crs(EPSG_5179, EPSG_WGS84, always_xy=True)


class BlockDivider:

    def divide(
        self,
        boundary_geojson: dict,
        road_polygons_gdf: gpd.GeoDataFrame
    ) -> dict:
        """
        도로로 경계 분할하여 블록 생성

        반환:
        {
          'blocks_gdf': GeoDataFrame,   # 블록 Polygon들
          'blocks_geojson': dict,
          'block_stats': dict
        }

        blocks_gdf columns:
        - geometry: Polygon
        - block_id: str ('B-001' 형식)
        - area_m2: float
        - adjacent_road_width_m: float  # 최대 접면도로폭
        - adjacent_road_count: int      # 접면도로 수
        - is_corner: bool               # 코너블록 여부
        - centroid_x: float
        - centroid_y: float
        """
        try:
            geom = shape(boundary_geojson)
            coords = list(geom.exterior.coords)
            transformed = [transformer_to_5179.transform(lon, lat) for lon, lat in coords]
            boundary_geom = Polygon(transformed)

            # Get road union
            if road_polygons_gdf.empty:
                road_union = None
            else:
                try:
                    road_union = unary_union(road_polygons_gdf.geometry.values)
                except Exception:
                    road_union = None

            # Subtract roads from boundary to get buildable area
            if road_union is not None and not road_union.is_empty:
                try:
                    buildable = boundary_geom.difference(road_union)
                except Exception:
                    buildable = boundary_geom
            else:
                buildable = boundary_geom

            # Extract individual polygons
            blocks = []
            if buildable.geom_type == 'Polygon':
                blocks = [buildable]
            elif buildable.geom_type == 'MultiPolygon':
                blocks = list(buildable.geoms)
            else:
                # Try to extract polygons from geometry collection
                for g in buildable.geoms if hasattr(buildable, 'geoms') else []:
                    if g.geom_type == 'Polygon' and g.area > 100:
                        blocks.append(g)

            # Filter out very small slivers
            blocks = [b for b in blocks if b.area > 500 and b.is_valid]

            # If no blocks, return boundary as single block
            if not blocks:
                blocks = [boundary_geom]

            # Build GeoDataFrame
            rows = []
            for i, block in enumerate(blocks):
                block_id = f'B-{i+1:03d}'
                area_m2 = block.area
                centroid = block.centroid

                adj_road_width = self._get_adjacent_road_width(block, road_polygons_gdf)
                adj_road_count = self._count_adjacent_roads(block, road_polygons_gdf)
                is_corner = self._is_corner_block(block, road_polygons_gdf)

                rows.append({
                    'geometry': block,
                    'block_id': block_id,
                    'area_m2': round(area_m2, 1),
                    'adjacent_road_width_m': adj_road_width,
                    'adjacent_road_count': adj_road_count,
                    'is_corner': is_corner,
                    'centroid_x': round(centroid.x, 1),
                    'centroid_y': round(centroid.y, 1),
                })

            blocks_gdf = gpd.GeoDataFrame(rows, crs=f"EPSG:{EPSG_5179}")

            # Calculate stats
            block_stats = self._calculate_stats(blocks_gdf)

            # Convert to WGS84 GeoJSON
            blocks_wgs84 = blocks_gdf.to_crs(EPSG_WGS84)
            blocks_geojson = json.loads(blocks_wgs84.to_json())

            return {
                'blocks_gdf': blocks_gdf,
                'blocks_geojson': blocks_geojson,
                'block_stats': block_stats,
            }

        except Exception as e:
            empty_gdf = gpd.GeoDataFrame(
                columns=['geometry', 'block_id', 'area_m2', 'adjacent_road_width_m',
                         'adjacent_road_count', 'is_corner', 'centroid_x', 'centroid_y'],
                crs=f"EPSG:{EPSG_5179}"
            )
            return {
                'blocks_gdf': empty_gdf,
                'blocks_geojson': {'type': 'FeatureCollection', 'features': []},
                'block_stats': {},
            }

    def _get_adjacent_road_width(
        self,
        block_geom,
        road_polygons_gdf: gpd.GeoDataFrame
    ) -> float:
        """
        블록에 접면한 도로 중 최대 폭원 반환
        접면 기준: 블록 경계와 1m 이내 도로
        도로 없으면 0.0 반환
        """
        if road_polygons_gdf.empty:
            return 0.0

        try:
            block_boundary = block_geom.exterior
            max_width = 0.0

            for idx, row in road_polygons_gdf.iterrows():
                road_geom = row.geometry
                if road_geom is None or road_geom.is_empty:
                    continue
                dist = block_boundary.distance(road_geom)
                if dist <= 1.0:
                    width = float(row.get('width_m', 0))
                    if width > max_width:
                        max_width = width

            return max_width
        except Exception:
            return 0.0

    def _count_adjacent_roads(
        self,
        block_geom,
        road_polygons_gdf: gpd.GeoDataFrame
    ) -> int:
        """접면 도로 개수 반환"""
        if road_polygons_gdf.empty:
            return 0

        try:
            block_boundary = block_geom.exterior
            count = 0

            for idx, row in road_polygons_gdf.iterrows():
                road_geom = row.geometry
                if road_geom is None or road_geom.is_empty:
                    continue
                dist = block_boundary.distance(road_geom)
                if dist <= 1.0:
                    count += 1

            return count
        except Exception:
            return 0

    def _is_corner_block(
        self,
        block_geom,
        road_polygons_gdf: gpd.GeoDataFrame
    ) -> bool:
        """
        2개 이상 도로에 접하면 코너블록
        """
        return self._count_adjacent_roads(block_geom, road_polygons_gdf) >= 2

    def _calculate_stats(self, blocks_gdf: gpd.GeoDataFrame) -> dict:
        """블록 통계"""
        if blocks_gdf.empty:
            return {
                'count': 0,
                'total_area_m2': 0,
                'avg_area_m2': 0,
                'min_area_m2': 0,
                'max_area_m2': 0,
                'corner_block_count': 0,
            }

        areas = blocks_gdf['area_m2']
        return {
            'count': len(blocks_gdf),
            'total_area_m2': round(float(areas.sum()), 1),
            'avg_area_m2': round(float(areas.mean()), 1),
            'min_area_m2': round(float(areas.min()), 1),
            'max_area_m2': round(float(areas.max()), 1),
            'corner_block_count': int(blocks_gdf['is_corner'].sum()),
        }
