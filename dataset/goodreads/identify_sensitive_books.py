import json
import os
import argparse
import re
from collections import defaultdict, Counter


# Predefined sensitive category keywords
SENSITIVE_CATEGORIES = {
    'mental_health': {
        'keywords': {
            'exact': ['suicide', 'depression', 'anxiety', 'trauma', 'ptsd'],
            'relaxed': [
                'mental health', 'mental illness', 'self harm', 'self-harm',
                'therapy', 'psychiatric', 'breakdown', 'bipolar', 'schizophrenia',
                'medication', 'overdose', 'cutting', 'grief', 'bereavement',
                'disorder', 'anorexia', 'bulimia', 'eating disorder'
            ]
        },
        'description': 'Mental health related content',
        'min_signals': 2  # Require title + description, or title + shelf, etc.
    },
    'violence': {
        'keywords': {
            'exact': ['murder', 'violence', 'war', 'torture', 'abuse', 'assault'],
            'relaxed': [
                'weapon', 'blood', 'kill', 'death', 'gun', 'shoot', 'attack',
                'revenge', 'massacre', 'slaughter', 'rape', 'sexual assault'
            ]
        },
        'description': 'Violent content and themes',
        'min_signals': 2
    },
    'explicit': {
        'keywords': {
            'exact': ['sex', 'erotic', 'porn', 'nude'],
            'relaxed': [
                'explicit', 'adult', 'sexual', 'xxx', 'nsfw', 'uncensored',
                'sensual', 'nudity', 'erotica'
            ]
        },
        'description': 'Explicit sexual content',
        'min_signals': 1
    }
}


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


def search_books_advanced(
    books_file='goodreads_books.json',
    category='mental_health',
    output_file='sensitive_books.txt',
    min_signals=None,
    use_title=True,
    use_description=True,
    use_shelves=True,
    genre_file=None
):
    """
    Search for books containing sensitive content using multi-signal detection.
    
    Args:
        books_file: Path to goodreads_books.json
        category: Predefined category name
        output_file: File to write sensitive book IDs to
        min_signals: Minimum number of signals required
        use_title: Whether to search in book titles
        use_description: Whether to search in book descriptions
        use_shelves: Whether to search in popular shelves
        genre_file: Optional path to genre file for additional signal
    """
    
    # Get category configuration
    if category in SENSITIVE_CATEGORIES:
        cat_config = SENSITIVE_CATEGORIES[category]
    else:
        print(f"Warning: Category '{category}' not found. Using 'mental_health' as default.")
        cat_config = SENSITIVE_CATEGORIES['mental_health']
    
    min_signals = min_signals if min_signals is not None else cat_config.get('min_signals', 2)
    
    if not os.path.exists(books_file):
        print(f"Error: Books file '{books_file}' not found!")
        return set()
    
    sensitive_books = {}  # book_id -> set of signals matched
    book_titles = {}  # book_id -> title (for reporting)
    
    # Compile regex patterns
    exact_patterns = []
    relaxed_patterns = []
    
    keywords = cat_config['keywords']
    if keywords.get('exact'):
        exact_pattern = '|'.join(r'\b' + re.escape(kw.lower()) + r'\b' for kw in keywords['exact'])
        exact_patterns = [re.compile(exact_pattern, re.IGNORECASE)]
    
    if keywords.get('relaxed'):
        relaxed_pattern = '|'.join(re.escape(kw.lower()) for kw in keywords['relaxed'])
        relaxed_patterns = [re.compile(relaxed_pattern, re.IGNORECASE)]
    
    # Load genre data if provided
    book_genres = {}
    if genre_file and os.path.exists(genre_file):
        book_genres = load_genre_data(genre_file)
    
    print(f"\nSearching for books in category: {category}")
    print(f"  Description: {cat_config['description']}")
    print(f"  Minimum signals required: {min_signals}")
    print(f"  Signals: title={use_title}, description={use_description}, shelves={use_shelves}, genres={bool(book_genres)}")
    print(f"Processing {books_file}...")
    
    count = 0
    with open(books_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                book_id = data.get('book_id')
                
                if not book_id:
                    continue
                
                signals = set()
                
                # Signal 1: Title
                if use_title:
                    title = data.get('title', '') or ''
                    if title:
                        book_titles[book_id] = title
                        title_lower = title.lower()
                        # Check exact patterns
                        for pattern in exact_patterns:
                            if pattern.search(title_lower):
                                signals.add('title_exact')
                                break
                        # Check relaxed patterns
                        if 'title_exact' not in signals:
                            for pattern in relaxed_patterns:
                                if pattern.search(title_lower):
                                    signals.add('title_relaxed')
                                    break
                
                # Signal 2: Description
                if use_description:
                    description = data.get('description', '') or ''
                    if description:
                        desc_lower = description.lower()
                        for pattern in exact_patterns:
                            if pattern.search(desc_lower):
                                signals.add('description_exact')
                                break
                        if 'description_exact' not in signals:
                            for pattern in relaxed_patterns:
                                if pattern.search(desc_lower):
                                    signals.add('description_relaxed')
                                    break
                
                # Signal 3: Popular shelves
                if use_shelves:
                    popular_shelves = data.get('popular_shelves', [])
                    for shelf in popular_shelves:
                        shelf_name = shelf.get('name', '').lower()
                        if shelf_name:
                            for pattern in exact_patterns:
                                if pattern.search(shelf_name):
                                    signals.add('shelf')
                                    break
                            if 'shelf' not in signals:
                                for pattern in relaxed_patterns:
                                    if pattern.search(shelf_name):
                                        signals.add('shelf')
                                        break
                        if 'shelf' in signals:
                            break
                
                # Signal 4: Genres (if available)
                if book_genres and book_id in book_genres:
                    genres = book_genres[book_id]
                    for genre_name in genres.keys():
                        genre_lower = genre_name.lower()
                        for pattern in exact_patterns:
                            if pattern.search(genre_lower):
                                signals.add('genre')
                                break
                        if 'genre' not in signals:
                            for pattern in relaxed_patterns:
                                if pattern.search(genre_lower):
                                    signals.add('genre')
                                    break
                        if 'genre' in signals:
                            break
                
                if signals:
                    sensitive_books[book_id] = signals
                    count += 1
            
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
    
    # Filter by minimum signals required
    filtered_books = {}
    signal_counts = Counter()
    
    for book_id, signals in sensitive_books.items():
        signal_count = len(signals)
        signal_counts[signal_count] += 1
        
        if signal_count >= min_signals:
            filtered_books[book_id] = signals
    
    # Print statistics
    print(f"\nSignal distribution:")
    for count in sorted(signal_counts.keys()):
        print(f"  {count} signal(s): {signal_counts[count]} books")
    
    print(f"\nTotal books matching criteria (min_signals={min_signals}): {len(filtered_books)}")
    
    if len(filtered_books) == 0:
        print("\nWarning: No sensitive books found!")
        print("Possible reasons:")
        print("  1. The keywords don't match any books")
        print("  2. The minimum signals threshold is too high")
        print("\nTry:")
        print("  - Lowering --min-signals")
        print("  - Using a different category")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for book_id in sorted(filtered_books):
                f.write(f"{book_id}\n")
        
        print(f"\nWritten sensitive book IDs to {output_file}")
        
        # Show sample matches
        print(f"\nSample matches (first 10):")
        for i, (book_id, signals) in enumerate(list(filtered_books.items())[:10]):
            title = book_titles.get(book_id, f"Book {book_id}")
            print(f"  {book_id}: {title[:60]:<60} [signals: {', '.join(sorted(signals))}]")
    
    return filtered_books


def search_books_by_keywords(
    books_file='goodreads_books.json',
    keywords=None,
    output_file='sensitive_books.txt',
    search_title=True,
    search_description=True,
    search_shelves=True
):
    """
    Legacy function for backward compatibility.
    Search for books containing specific keywords in their metadata.
    """
    if not keywords:
        print("Error: No keywords provided!")
        return set()

    if not os.path.exists(books_file):
        print(f"Error: Books file '{books_file}' not found!")
        return set()

    sensitive_books = set()
    matched_info = defaultdict(list)

    keyword_patterns = [
        (keyword, re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE))
        for keyword in keywords
    ]

    print(f"\nSearching for books with keywords: {keywords}")
    print(f"Search in: title={search_title}, description={search_description}, shelves={search_shelves}")
    print(f"Processing {books_file}...")

    count = 0
    with open(books_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                book_id = data.get('book_id')

                if not book_id:
                    continue

                matched = False
                match_locations = []

                if search_title:
                    title = data.get('title', '') or ''
                    for keyword, pattern in keyword_patterns:
                        if pattern.search(title):
                            matched = True
                            match_locations.append(f"title (keyword: {keyword})")
                            break

                if search_description and not matched:
                    description = data.get('description', '') or ''
                    for keyword, pattern in keyword_patterns:
                        if pattern.search(description):
                            matched = True
                            match_locations.append(f"description (keyword: {keyword})")
                            break

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

    print(f"\nTotal unique sensitive books found: {len(sensitive_books)}")

    if len(sensitive_books) == 0:
        print("\nWarning: No sensitive books found!")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for book_id in sorted(sensitive_books):
                f.write(f"{book_id}\n")

        print(f"Written sensitive book IDs to {output_file}")

    return sensitive_books


def search_books_by_genre(
    genre_file='goodreads_book_genres_initial.json',
    genre_keywords=None,
    output_file='sensitive_books.txt'
):
    """
    Legacy function for backward compatibility.
    Search for books by genre keywords.
    """
    if not genre_keywords:
        print("Error: No genre keywords provided!")
        return set()

    if not os.path.exists(genre_file):
        print(f"Error: Genre file '{genre_file}' not found!")
        return set()

    sensitive_books = set()

    genre_patterns = [
        (keyword, re.compile(re.escape(keyword.lower()), re.IGNORECASE))
        for keyword in genre_keywords
    ]

    print(f"\nSearching for books with genres: {genre_keywords}")
    book_genres = load_genre_data(genre_file)

    for book_id, genres in book_genres.items():
        for genre_name in genres.keys():
            for keyword, pattern in genre_patterns:
                if pattern.search(genre_name):
                    sensitive_books.add(book_id)
                    break
            if book_id in sensitive_books:
                break

    print(f"Total unique sensitive books: {len(sensitive_books)}")

    if len(sensitive_books) > 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            for book_id in sorted(sensitive_books):
                f.write(f"{book_id}\n")
        print(f"Written sensitive book IDs to {output_file}")

    return sensitive_books


def main():
    parser = argparse.ArgumentParser(
        description='Identify sensitive books from Goodreads dataset using multi-signal detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined category with multi-signal detection (RECOMMENDED)
  python identify_sensitive_books.py --category mental_health --output sensitive_books_health.txt
  
  # Use violence category
  python identify_sensitive_books.py --category violence --output sensitive_books_violence.txt
  
  # Customize minimum signals required
  python identify_sensitive_books.py --category mental_health --min-signals 3 --output sensitive_books_strict.txt
  
  # Disable specific signals
  python identify_sensitive_books.py --category mental_health --no-shelves --output sensitive_books.txt
  
  # Legacy mode: Use custom keywords
  python identify_sensitive_books.py --keywords suicide depression --output sensitive_books_health.txt
  
  # Legacy mode: Use genres
  python identify_sensitive_books.py --use-genres --genres romance erotica --output sensitive_books_romance.txt

Available categories:
  - mental_health: Mental health related content (requires 2+ signals)
  - violence: Violent content and themes (requires 2+ signals)
  - explicit: Explicit sexual content (requires 1+ signal)

Important:
  - Multi-signal detection reduces false positives by requiring matches in multiple fields
  - Use --category for better results than --keywords
        """
    )
    
    # New advanced mode arguments
    parser.add_argument(
        '--category',
        choices=list(SENSITIVE_CATEGORIES.keys()),
        default=None,
        help='Predefined sensitive category to use (mental_health, violence, explicit)'
    )
    parser.add_argument(
        '--min-signals',
        type=int,
        default=None,
        help='Minimum number of signals required (title, description, shelves, genres). Overrides category default.'
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
    
    # Legacy mode arguments
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=None,
        help='[Legacy] Keywords to search for in book metadata'
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
    
    # Alternative mode: genre search
    parser.add_argument(
        '--use-genres',
        action='store_true',
        help='[Legacy] Use genre-based search instead of keyword search (requires --genres)'
    )
    parser.add_argument(
        '--genres',
        nargs='+',
        default=None,
        help='[Legacy] Genre keywords for genre-based search (only with --use-genres)'
    )
    parser.add_argument(
        '--genre-file',
        default='goodreads_book_genres_initial.json',
        help='Path to genre JSON file (default: goodreads_book_genres_initial.json)'
    )

    args = parser.parse_args()

    # Determine which mode to use
    if args.category:
        # Advanced multi-signal mode
        search_books_advanced(
            books_file=args.books_file,
            category=args.category,
            output_file=args.output,
            min_signals=args.min_signals,
            use_title=not args.no_title,
            use_description=not args.no_description,
            use_shelves=not args.no_shelves,
            genre_file=args.genre_file if args.genre_file != 'goodreads_book_genres_initial.json' else None
        )
    elif args.use_genres:
        # Legacy genre-based search
        if not args.genres:
            print("Error: --use-genres requires --genres to be specified")
            return
        search_books_by_genre(
            genre_file=args.genre_file,
            genre_keywords=args.genres,
            output_file=args.output
        )
    elif args.keywords:
        # Legacy keyword-based search
        search_books_by_keywords(
            books_file=args.books_file,
            keywords=args.keywords,
            output_file=args.output,
            search_title=not args.no_title,
            search_description=not args.no_description,
            search_shelves=not args.no_shelves
        )
    else:
        print("Error: Must specify either --category, --keywords, or --use-genres")
        print("\nRecommended: Use --category for better results")
        print("Example: python identify_sensitive_books.py --category mental_health")
        return


if __name__ == "__main__":
    main()
