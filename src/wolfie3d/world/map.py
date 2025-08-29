#!/usr/bin/env python3
"""
Map and collision handling for Wolfie3D game
"""

from ..utils.constants import MAP, MAP_W, MAP_H, PLAYER_START_X, PLAYER_START_Y
import random


def in_map(ix: int, iy: int) -> bool:
    """Check if coordinates are within map bounds."""
    return 0 <= ix < MAP_W and 0 <= iy < MAP_H


def is_wall(ix: int, iy: int) -> bool:
    """Check if position contains a wall."""
    return in_map(ix, iy) and MAP[iy][ix] > 0


def is_floor(ix: int, iy: int) -> bool:
    """Check if position is floor (not wall)."""
    return in_map(ix, iy) and MAP[iy][ix] == 0


def find_accessible_areas() -> set[tuple[int, int]]:
    """
    Find all floor positions that are accessible from the player's starting position.
    Uses flood-fill algorithm to find connected areas.
    
    Returns:
        set: Set of (x, y) coordinates that are accessible
    """
    start_x, start_y = int(PLAYER_START_X), int(PLAYER_START_Y)
    accessible = set()
    to_visit = [(start_x, start_y)]
    
    while to_visit:
        x, y = to_visit.pop()
        
        if (x, y) in accessible or not is_floor(x, y):
            continue
            
        accessible.add((x, y))
        
        # Check all 4 directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if is_floor(nx, ny) and (nx, ny) not in accessible:
                to_visit.append((nx, ny))
    
    return accessible


def find_valid_spawn_position(existing_positions: list[tuple[float, float]] = None, min_distance: float = 2.0) -> tuple[float, float]:
    """
    Find a random valid spawn position on the floor, ensuring minimum distance from existing positions.
    Only spawns in areas accessible to the player.
    
    Args:
        existing_positions: List of existing positions to avoid
        min_distance: Minimum distance from existing positions
        
    Returns:
        tuple: (x, y) coordinates for a valid spawn position
    """
    if existing_positions is None:
        existing_positions = []
    
    # Get all accessible floor positions
    accessible_areas = find_accessible_areas()
    valid_positions = []
    
    # Check each accessible position
    for x, y in accessible_areas:
        # Add some random offset within the cell to avoid spawning exactly on grid lines
        pos = (x + 0.5, y + 0.5)
        
        # Check distance from existing positions
        too_close = False
        for existing_pos in existing_positions:
            dx = pos[0] - existing_pos[0]
            dy = pos[1] - existing_pos[1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            valid_positions.append(pos)
    
    if not valid_positions:
        # Fallback to a safe position if no valid positions found
        return (1.5, 1.5)
    
    return random.choice(valid_positions)


def try_move(nx: float, ny: float, current_x: float, current_y: float) -> tuple[float, float]:
    """
    Try to move to new position, handling wall collisions.
    
    Args:
        nx: New x position
        ny: New y position
        current_x: Current x position
        current_y: Current y position
        
    Returns:
        tuple: (final_x, final_y) - actual position after collision handling
    """
    if not is_wall(int(nx), int(current_y)):
        x = nx
    else:
        x = current_x
    if not is_wall(int(x), int(ny)):
        y = ny
    else:
        y = current_y
    return x, y


def clamp01(x: float) -> float:
    """Clamp value between 0.0 and 1.0."""
    if x < 0.0: 
        return 0.0
    if x > 1.0: 
        return 1.0
    return x


def debug_print_accessible_areas():
    """
    Debug function to print the map with accessible areas marked.
    Useful for verifying that the flood-fill algorithm works correctly.
    """
    accessible = find_accessible_areas()
    
    print("Map with accessible areas (A=accessible, W=wall, I=inaccessible floor):")
    for y in range(MAP_H):
        row = ""
        for x in range(MAP_W):
            if is_wall(x, y):
                row += "W"
            elif (x, y) in accessible:
                row += "A"
            else:
                row += "I"
        print(row)
    
    print(f"Total accessible positions: {len(accessible)}")
    print(f"Player start position: ({int(PLAYER_START_X)}, {int(PLAYER_START_Y)})")
