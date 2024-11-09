import os
import pandas as pd
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from fuzzywuzzy import fuzz
from datetime import datetime

def get_mp3_metadata(file_path):
    """Extract the title, year, and artist metadata from an MP3 file."""
    try:
        audio = MP3(file_path, ID3=EasyID3)
        title = audio.get('title', ['Unknown Title'])[0]
        date = audio.get('date', ['Unknown Year'])[0]
        year = int(date.split('-')[0])  # Assume 'metadata_year' might include more details
        artist = audio.get('artist', ['Unknown Artist'])[0]
        return title, year, artist
    except Exception:
        return None, None, None

def get_file_path_for_title_and_year_artist(title, year, artist, folder_path, threshold):
    mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
    best_match = None
    best_score = 0

    for file in mp3_files:
        file_path = os.path.join(folder_path, file)
        metadata_title, metadata_year, metadata_artist = get_mp3_metadata(file_path)

        if metadata_title and metadata_year:
            if threshold > 60:
                title_score = fuzz.ratio(title, metadata_title)
                if (title_score*0.85) < threshold:
                    try:
                        year_int = int(year)
                        year_score = (year_int == metadata_year) * 100
                    except ValueError:
                        year_score = (str(year) in metadata_year) * 100
                    combined_score = (title_score + (year_score * 0.7)) // 2
                    # print(f"{title}, FILE: {metadata_title} titleScore: {title_score}, year_score {year_score} and combined:{combined_score}")
                else:
                    combined_score = title_score
                    # print(f"{title}, FILE: {metadata_title} titleScore: {title_score} and combined:{combined_score}")

            if threshold < 60 and metadata_artist:
                title_score = fuzz.token_set_ratio(title, metadata_title)
                if (title_score * 0.75) < threshold:
                    try:
                        year_int = int(year)
                        year_score = (year_int == metadata_year) * 100
                    except ValueError:
                        year_score = (str(year) in metadata_year) * 100
                    combined_score = (title_score + (year_score * 0.7)) // 2
                    # print(f"{title}, FILE: {metadata_title} titleScore: {title_score}, year_score {year_score} and combined:{combined_score}")
                else:
                    combined_score = title_score
                    # print(f"{title}, FILE: {metadata_title} titleScore: {title_score} and combined:{combined_score}")
                artist_score = fuzz.partial_ratio(artist, metadata_artist)
                combined_score = (combined_score + (artist_score * 0.6)) // 2

            if combined_score > best_score:
                best_match = file
                best_score = combined_score

    if best_match and best_score >= threshold:
        print(f"Match found: for {title} ({year}) by {artist}: {best_match} with a score of {best_score}%")
        return os.path.join(folder_path, best_match), best_score
    else:
        print(f"No Match found: for {title} ({year}) by {artist}: probable {best_match} is with a score of {best_score}%")
        return 'null', best_score

def add_file_path_column(csv_file_path, folder_path):
    df = pd.read_csv(csv_file_path)

    if 'SongTitle' not in df.columns or 'ReleaseYear' not in df.columns or 'Artist' not in df.columns:
        print("The CSV must have 'SongTitle', 'ReleaseYear', and 'Artist' columns.")
        return

    df['file_path'] = 'null'
    df['score'] = None
    df['threshold'] = None

    thresholds = [75, 65, 50]

    for threshold in thresholds:
        print(threshold)
        pending_indices = df[df['file_path'] == 'null'].index
        pending_titles = df.loc[pending_indices, 'SongTitle'].tolist()
        pending_years = df.loc[pending_indices, 'ReleaseYear'].tolist()
        pending_artists = df.loc[pending_indices, 'Artist'].tolist()

        updated_results = []
        for title, year, artist in zip(pending_titles, pending_years, pending_artists):
            result = get_file_path_for_title_and_year_artist(title, year, artist, folder_path, threshold)
            updated_results.append(result)

        updated_file_paths, used_thresholds = zip(*updated_results)
        df.loc[pending_indices, 'file_path'] = updated_file_paths
        df.loc[pending_indices, 'score'] = used_thresholds
        df.loc[pending_indices, 'threshold'] = threshold

        if not df['file_path'].str.contains('null').any():
            break

    output_file_path = 'updated_File_multi' + os.path.basename(csv_file_path)
    df.to_csv(output_file_path, index=False)
    print(f"Updated CSV saved as: {output_file_path}")

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
csv_file_path = 'cleaned_genres_2.csv'
folder_path = '../../audio_data/Billboard'
add_file_path_column(csv_file_path, folder_path)


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))