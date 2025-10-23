import csv
import os
import argparse
import re


def extract_sensitive_movies_by_genre(
    movie_file='movie.csv',
    tag_file='tag.csv',
    genres=None,
    tags=None,
    output_file='sensitive_movies.txt',
    use_genres=True,
    use_tags=False
):
    """
    Extract movie IDs based on sensitive genres or user tags.

    Args:
        movie_file: Path to movie.csv (movieId, title, genres)
        tag_file: Path to tag.csv (userId, movieId, tag, timestamp)
        genres: List of genre keywords to search for (case-insensitive, whole-word matching)
        tags: List of tag keywords to search for (case-insensitive)
        output_file: File to write sensitive movie IDs to
        use_genres: Whether to search in movie genres
        use_tags: Whether to search in user tags
    """

    sensitive_movies = set()

    # Compile regex patterns for whole-word matching (case-insensitive)
    if genres:
        genre_patterns = [re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
                          for keyword in genres]
    else:
        genre_patterns = []

    if tags:
        tag_patterns = [re.compile(re.escape(keyword.lower()), re.IGNORECASE)
                        for keyword in tags]
    else:
        tag_patterns = []

    # Search in movie genres
    if use_genres and genre_patterns and os.path.exists(movie_file):
        print(f"Searching for movies with genres: {genres}")
        print(f"Processing {movie_file}...")

        genre_count = 0
        with open(movie_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id = row['movieId']
                genre_str = row['genres']

                # Check if any sensitive genre keyword is in the genres
                for pattern in genre_patterns:
                    if pattern.search(genre_str):
                        sensitive_movies.add(movie_id)
                        genre_count += 1
                        break

        print(f"  Found {genre_count} movies with sensitive genres")

    # Search in user tags
    if use_tags and tag_patterns and os.path.exists(tag_file):
        print(f"Searching for movies with tags: {tags}")
        print(f"Processing {tag_file}...")

        tag_count = 0
        with open(tag_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id = row['movieId']
                tag_str = row['tag']

                # Check if any sensitive tag keyword is in the tag
                for pattern in tag_patterns:
                    if pattern.search(tag_str):
                        if movie_id not in sensitive_movies:
                            sensitive_movies.add(movie_id)
                            tag_count += 1
                        break

        print(f"  Found {tag_count} additional movies with sensitive tags")

    # Write to output file
    print(f"\nTotal unique sensitive movies: {len(sensitive_movies)}")

    if len(sensitive_movies) == 0:
        print("\nWarning: No sensitive movies found!")
        print("Possible reasons:")
        print("  1. The genre/tag keywords don't match any movies")
        print("  2. The movie.csv or tag.csv files are not in the expected format")
        print("\nMake sure to:")
        print("  - Use appropriate genre/tag keywords")
        print("  - Check that movie.csv and tag.csv exist in the current directory")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for movie_id in sorted(sensitive_movies, key=int):
                f.write(f"{movie_id}\n")

        print(f"Written sensitive movie IDs to {output_file}")

    return sensitive_movies


def main():
    parser = argparse.ArgumentParser(
        description='Identify sensitive movies from MovieLens dataset based on genres or tags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find horror movies by genre
  python identify_sensitive_movies.py --genres Horror --output sensitive_movies_horror.txt

  # Find movies with multiple genres
  python identify_sensitive_movies.py --genres Horror Thriller --output sensitive_movies_horror_thriller.txt

  # Find movies by user tags
  python identify_sensitive_movies.py --tags "violent" "disturbing" --use-tags --output sensitive_movies_violent.txt

  # Find movies by both genres and tags
  python identify_sensitive_movies.py --genres Horror --tags "scary" "disturbing" --use-tags --output sensitive_movies_horror_all.txt

  # Find romance movies
  python identify_sensitive_movies.py --genres Romance --output sensitive_movies_romance.txt

  # Find sci-fi movies
  python identify_sensitive_movies.py --genres "Sci-Fi" --output sensitive_movies_scifi.txt

Common MovieLens genres:
  Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama,
  Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller,
  War, Western, IMAX
        """
    )
    parser.add_argument(
        '--genres',
        nargs='+',
        default=None,
        help='Genre keywords to search for (case-insensitive, whole-word matching)'
    )
    parser.add_argument(
        '--tags',
        nargs='+',
        default=None,
        help='Tag keywords to search for in user tags (case-insensitive, partial matching)'
    )
    parser.add_argument(
        '--output',
        default='sensitive_movies.txt',
        help='Output file for sensitive movie IDs (default: sensitive_movies.txt)'
    )
    parser.add_argument(
        '--movie-file',
        default='movie.csv',
        help='Path to movie.csv (default: movie.csv)'
    )
    parser.add_argument(
        '--tag-file',
        default='tag.csv',
        help='Path to tag.csv (default: tag.csv)'
    )
    parser.add_argument(
        '--use-tags',
        action='store_true',
        help='Search in user tags (requires --tags)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.genres and not (args.tags and args.use_tags):
        print("Error: Must specify either --genres or both --tags and --use-tags")
        return

    if not os.path.exists(args.movie_file) and args.genres:
        print(f"Error: Movie file '{args.movie_file}' not found!")
        return

    if not os.path.exists(args.tag_file) and args.use_tags:
        print(f"Warning: Tag file '{args.tag_file}' not found!")
        print("Continuing with genre search only...")
        args.use_tags = False

    extract_sensitive_movies_by_genre(
        movie_file=args.movie_file,
        tag_file=args.tag_file,
        genres=args.genres,
        tags=args.tags,
        output_file=args.output,
        use_genres=args.genres is not None,
        use_tags=args.use_tags
    )


if __name__ == "__main__":
    main()
