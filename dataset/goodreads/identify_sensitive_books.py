import json
import os
import argparse
import re
from collections import defaultdict


def load_genre_data(genre_file):
    """
    Load genre information from goodreads_book_genres_initial.json.
    Returns: dict mapping book_id to dict of genres with their counts
    """
    print(f"Loading genre data from {genre_file}...")
    book_genres = {}

    with open(genre_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                book_id = data.get('book_id')
                genres = data.get('genres', {})
                if book_id and genres:
                    book_genres[book_id] = genres
            except:
                continue

    print(f"Loaded genre data for {len(book_genres)} books")
    return book_genres


def search_books_by_keywords(
    books_file='goodreads_books.json',
    keywords=None,
    output_file='sensitive_books.txt',
    search_title=True,
    search_description=True,
    search_shelves=True
):
    """
    Search for books containing specific keywords in their metadata.

    Args:
        books_file: Path to goodreads_books.json
        keywords: List of keywords to search for (case-insensitive, whole-word matching)
        output_file: File to write sensitive book IDs to
        search_title: Whether to search in book titles
        search_description: Whether to search in book descriptions
        search_shelves: Whether to search in popular shelves
    """

    if not keywords:
        print("Error: No keywords provided!")
        return set()

    if not os.path.exists(books_file):
        print(f"Error: Books file '{books_file}' not found!")
        return set()

    sensitive_books = set()
    matched_info = defaultdict(list)  # Track what matched for each book

    # Compile regex patterns for whole-word matching (case-insensitive)
    # \b ensures we match whole words only
    keyword_patterns = [
        (keyword, re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE))
        for keyword in keywords
    ]

    print(f"\nSearching for books with keywords: {keywords}")
    print(f"Search in: title={search_title}, description={search_description}, shelves={search_shelves}")
    print(f"Processing {books_file}...")

    count = 0
    with open(books_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                book_id = data.get('book_id')

                if not book_id:
                    continue

                matched = False
                match_locations = []

                # Search in title
                if search_title:
                    title = data.get('title', '') or ''
                    for keyword, pattern in keyword_patterns:
                        if pattern.search(title):
                            matched = True
                            match_locations.append(f"title (keyword: {keyword})")
                            break

                # Search in description
                if search_description and not matched:
                    description = data.get('description', '') or ''
                    for keyword, pattern in keyword_patterns:
                        if pattern.search(description):
                            matched = True
                            match_locations.append(f"description (keyword: {keyword})")
                            break

                # Search in popular shelves
                if search_shelves and not matched:
                    popular_shelves = data.get('popular_shelves', [])
                    for shelf in popular_shelves:
                        shelf_name = shelf.get('name', '')
                        for keyword, pattern in keyword_patterns:
                            if pattern.search(shelf_name):
                                matched = True
                                match_locations.append(f"shelf '{shelf_name}' (keyword: {keyword})")
                                break
                        if matched:
                            break

                if matched:
                    sensitive_books.add(book_id)
                    matched_info[book_id] = match_locations
                    count += 1

            except json.JSONDecodeError:
                continue
            except Exception:
                continue

            if line_num % 100000 == 0:
                print(f"  Processed {line_num} books, found {count} matching books so far...")

    print(f"\nTotal unique sensitive books found: {len(sensitive_books)}")

    if len(sensitive_books) == 0:
        print("\nWarning: No sensitive books found!")
        print("Possible reasons:")
        print("  1. The keywords don't match any books")
        print("  2. The keywords are too specific or misspelled")
        print("\nTry:")
        print("  - Using more general keywords")
        print("  - Using related terms or synonyms")
        print("  - Checking the spelling of your keywords")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for book_id in sorted(sensitive_books):
                f.write(f"{book_id}\n")

        print(f"Written sensitive book IDs to {output_file}")

        # Print sample of matches
        print(f"\nSample of matched books (first 10):")
        for i, book_id in enumerate(sorted(sensitive_books)[:10]):
            print(f"  Book ID {book_id}: matched in {', '.join(matched_info[book_id])}")

    return sensitive_books


def search_books_by_genre(
    genre_file='goodreads_book_genres_initial.json',
    genre_keywords=None,
    output_file='sensitive_books.txt'
):
    """
    Search for books by genre keywords (alternative method).
    """

    if not genre_keywords:
        print("Error: No genre keywords provided!")
        return set()

    if not os.path.exists(genre_file):
        print(f"Error: Genre file '{genre_file}' not found!")
        return set()

    sensitive_books = set()

    # Compile regex patterns for matching (case-insensitive)
    genre_patterns = [
        (keyword, re.compile(re.escape(keyword.lower()), re.IGNORECASE))
        for keyword in genre_keywords
    ]

    print(f"\nSearching for books with genres: {genre_keywords}")
    book_genres = load_genre_data(genre_file)

    genre_count = 0
    for book_id, genres in book_genres.items():
        # genres is a dict like {"fiction": 219, "romance": 45}
        for genre_name in genres.keys():
            for keyword, pattern in genre_patterns:
                if pattern.search(genre_name):
                    sensitive_books.add(book_id)
                    genre_count += 1
                    break
            if book_id in sensitive_books:
                break

    print(f"  Found {genre_count} books with sensitive genres")
    print(f"Total unique sensitive books: {len(sensitive_books)}")

    if len(sensitive_books) > 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            for book_id in sorted(sensitive_books):
                f.write(f"{book_id}\n")
        print(f"Written sensitive book IDs to {output_file}")

    return sensitive_books


def main():
    parser = argparse.ArgumentParser(
        description='Identify sensitive books from Goodreads dataset based on keywords or genres',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find books about mental health and suicide (DEFAULT MODE - keyword search)
  python identify_sensitive_books.py --keywords suicide "mental health" depression --output sensitive_books_mental_health.txt

  # Search only in titles
  python identify_sensitive_books.py --keywords suicide depression --no-description --no-shelves --output sensitive_books_suicide.txt

  # Search only in descriptions (most comprehensive)
  python identify_sensitive_books.py --keywords suicide "self-harm" --no-title --no-shelves --output sensitive_books_suicide.txt

  # Find books about specific sensitive topics
  python identify_sensitive_books.py --keywords abuse trauma "sexual assault" --output sensitive_books_trauma.txt
  python identify_sensitive_books.py --keywords addiction alcoholism "drug abuse" --output sensitive_books_addiction.txt
  python identify_sensitive_books.py --keywords eating disorder anorexia bulimia --output sensitive_books_eating_disorders.txt

  # ALTERNATIVE: Find books by genre (use --use-genres flag)
  python identify_sensitive_books.py --use-genres --genres romance erotica --output sensitive_books_romance.txt
  python identify_sensitive_books.py --use-genres --genres horror thriller --output sensitive_books_horror.txt

Keyword Search Tips:
  - Use quotes for multi-word phrases: "mental health", "eating disorder"
  - Keywords use whole-word matching (e.g., 'suicide' won't match 'suicidal')
  - Be specific but not too narrow
  - Try related terms: suicide, depression, "mental illness", "mental health"

Common Sensitive Topics:
  - Mental health: suicide, depression, anxiety, "mental health", "mental illness"
  - Trauma: abuse, trauma, "sexual assault", rape, violence
  - Substance abuse: addiction, alcoholism, "drug abuse", substance
  - Eating disorders: anorexia, bulimia, "eating disorder"
  - Self-harm: "self-harm", "self-injury", cutting
        """
    )
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=None,
        help='Keywords to search for in book metadata (default mode)'
    )
    parser.add_argument(
        '--output',
        default='sensitive_books.txt',
        help='Output file for sensitive book IDs (default: sensitive_books.txt)'
    )
    parser.add_argument(
        '--books-file',
        default='goodreads_books.json',
        help='Path to books JSON file (default: goodreads_books.json)'
    )
    parser.add_argument(
        '--no-title',
        action='store_true',
        help='Do not search in book titles'
    )
    parser.add_argument(
        '--no-description',
        action='store_true',
        help='Do not search in book descriptions'
    )
    parser.add_argument(
        '--no-shelves',
        action='store_true',
        help='Do not search in popular shelves'
    )

    # Alternative mode: genre search
    parser.add_argument(
        '--use-genres',
        action='store_true',
        help='Use genre-based search instead of keyword search (requires --genres)'
    )
    parser.add_argument(
        '--genres',
        nargs='+',
        default=None,
        help='Genre keywords for genre-based search (only with --use-genres)'
    )
    parser.add_argument(
        '--genre-file',
        default='goodreads_book_genres_initial.json',
        help='Path to genre JSON file (default: goodreads_book_genres_initial.json)'
    )

    args = parser.parse_args()

    # Determine which mode to use
    if args.use_genres:
        # Genre-based search
        if not args.genres:
            print("Error: --use-genres requires --genres to be specified")
            return
        search_books_by_genre(
            genre_file=args.genre_file,
            genre_keywords=args.genres,
            output_file=args.output
        )
    else:
        # Keyword-based search (default)
        if not args.keywords:
            print("Error: Must specify --keywords for keyword search (or use --use-genres for genre search)")
            return
        search_books_by_keywords(
            books_file=args.books_file,
            keywords=args.keywords,
            output_file=args.output,
            search_title=not args.no_title,
            search_description=not args.no_description,
            search_shelves=not args.no_shelves
        )


if __name__ == "__main__":
    main()
