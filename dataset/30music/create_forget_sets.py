#!/usr/bin/env python3
"""
Create forget sets for sensitive categories in 30music dataset

This script:
1. Identifies tracks with sensitive tags (health, explicit)
2. Finds all interactions with those tracks in the .inter file
3. Creates separate forget set files for each category
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Define sensitive category mappings
SENSITIVE_CATEGORIES = {
    'health': {
        'tag_ids': {'154395', '15345', '233689', '214036', '59294'},
        'tags': ['mental', 'anxiety', 'suicide', 'self-harm', 'depression'],
        'description': 'Mental health related content that may be triggering'
    },
    'explicit': {
        'tag_ids': {'77138', '193026', '174963'},
        'tags': ['explicit', 'profanity', 'nsfw'],
        'description': 'Explicit content with profanity or mature themes'
    }
}

def identify_sensitive_tracks(tracks_file, categories):
    """
    Identify tracks that have sensitive tags

    Args:
        tracks_file: Path to tracks.idomaar file
        categories: Dictionary of sensitive categories with tag_ids

    Returns:
        Dictionary mapping category name to set of track IDs
    """
    print("Identifying sensitive tracks from tags")

    # Initialize sets for each category
    sensitive_tracks = {cat: set() for cat in categories}

    print(f"\nProcessing tracks from: {tracks_file}")

    total_tracks = 0
    with open(tracks_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if i % 500000 == 0:
                print(f"  Processed {i:,} tracks...")

            total_tracks += 1
            parts = line.strip().split('\t')

            if len(parts) < 5:
                continue

            track_id = parts[1]

            try:
                relations = json.loads(parts[4])

                if 'tags' in relations and relations['tags']:
                    for tag in relations['tags']:
                        tag_id = str(tag['id'])

                        # Check each category
                        for cat_name, cat_info in categories.items():
                            if tag_id in cat_info['tag_ids']:
                                sensitive_tracks[cat_name].add(track_id)
            except:
                pass

    print(f"\nProcessed {total_tracks:,} total tracks")
    print("\nSensitive tracks found:")
    for cat_name, track_set in sensitive_tracks.items():
        print(f"  {cat_name}: {len(track_set):,} tracks")

    return sensitive_tracks

def create_forget_sets(inter_file, sensitive_tracks, output_dir, seeds=[2, 3, 5, 7, 11],
                       fractions=[1e-6, 1e-5, 1e-4]):
    """
    Create forget set files by sampling interactions with sensitive tracks

    Args:
        inter_file: Path to .inter file
        sensitive_tracks: Dictionary mapping category to set of sensitive track IDs
        output_dir: Directory to save forget set files
        seeds: List of random seeds for sampling
        fractions: List of unlearning fractions to sample
    """
    print("\nCreating forget sets")

    # First, load all interactions and identify which involve sensitive tracks
    print(f"\nLoading interactions from: {inter_file}")

    # Store interactions by category
    category_interactions = defaultdict(list)
    total_interactions = 0

    with open(inter_file, 'r') as f:
        header = f.readline().strip()

        for i, line in enumerate(f, 1):
            if i % 1000000 == 0:
                print(f"  Processed {i:,} interactions...")

            total_interactions += 1
            parts = line.strip().split('\t')

            if len(parts) < 4:
                continue

            user_id, session_id, item_id, timestamp = parts

            # Check if this interaction involves a sensitive track
            for cat_name, track_set in sensitive_tracks.items():
                if item_id in track_set:
                    category_interactions[cat_name].append(line.strip())

    print(f"\nTotal interactions: {total_interactions:,}")
    print("\nInteractions with sensitive content:")
    for cat_name, interactions in category_interactions.items():
        print(f"  {cat_name}: {len(interactions):,} interactions")
        print(f"    ({len(interactions)/total_interactions*100:.4f}% of total)")

    # Create forget sets for each category, seed, and fraction
    print("\nGenerating forget set files")

    import random

    for cat_name, interactions in category_interactions.items():
        if len(interactions) == 0:
            print(f"\nSkipping {cat_name}: no interactions found")
            continue

        print(f"\n{cat_name.upper()}:")
        print(f"  Total interactions: {len(interactions):,}")

        for seed in seeds:
            for fraction in fractions:
                # Calculate sample size based on total dataset size
                sample_size = int(total_interactions * fraction)

                if sample_size == 0:
                    print(f"  Skipping seed={seed}, fraction={fraction}: sample size = 0")
                    continue

                if sample_size > len(interactions):
                    print(f"  Skipping seed={seed}, fraction={fraction}: sample size ({sample_size}) exceeds available sensitive interactions ({len(interactions)})")
                    continue

                # Sample interactions
                random.seed(seed)
                sampled = random.sample(interactions, sample_size)

                # Create output filename
                output_file = output_dir / f"30music_unlearn_pairs_sensitive_category_{cat_name}_seed_{seed}_unlearning_fraction_{fraction}.inter"

                # Write forget set
                with open(output_file, 'w') as f:
                    f.write(header + '\n')
                    for interaction in sampled:
                        f.write(interaction + '\n')

                print(f"  Created: seed={seed}, fraction={fraction}, size={sample_size:,} -> {output_file.name}")

    print("\nForget set generation complete!")

def main():
    # Set paths
    base_dir = Path(__file__).parent
    tracks_file = base_dir / 'entities' / 'tracks.idomaar'
    inter_file = base_dir / '30music.inter'
    output_dir = base_dir

    # Check files exist
    if not tracks_file.exists():
        print(f"Error: Tracks file not found: {tracks_file}")
        sys.exit(1)

    if not inter_file.exists():
        print(f"Error: .inter file not found: {inter_file}")
        print("Please run convert_sessions_to_inter.py first")
        sys.exit(1)

    print("30MUSIC FORGET SET GENERATOR\n")
    print("Sensitive categories:")
    for cat_name, cat_info in SENSITIVE_CATEGORIES.items():
        print(f"\n{cat_name}:")
        print(f"  Tags: {', '.join(cat_info['tags'])}")
        print(f"  Description: {cat_info['description']}")

    # Step 1: Identify sensitive tracks
    sensitive_tracks = identify_sensitive_tracks(tracks_file, SENSITIVE_CATEGORIES)

    # Save sensitive track lists
    # Use 'sensitive_asins' naming for consistency with other datasets
    print("\nSaving sensitive track IDs...")
    for cat_name, track_set in sensitive_tracks.items():
        output_file = output_dir / f"sensitive_asins_{cat_name}.txt"
        with open(output_file, 'w') as f:
            for track_id in sorted(track_set, key=lambda x: int(x)):
                f.write(f"{track_id}\n")
        print(f"  {cat_name}: {output_file}")

    # Step 2: Create forget sets
    create_forget_sets(inter_file, sensitive_tracks, output_dir)

if __name__ == "__main__":
    main()