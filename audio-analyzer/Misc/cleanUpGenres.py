import pandas as pd
popular_genres = {
    "Pop": ["Pop", "R&B/Soul", "Reggae", "Electropop", "Synth-pop", "Teen pop", "Soul"],
    "Rock": ["Rock", "Alternative", "Indie", "Alternative/Indie", "Classic rock", "Hard rock", "Punk rock", "Metal", "Emo"],
    "Hip Hop": ["Hip-Hop", "Rap", "Hip-Hop/Rap", "Trap", "Gangsta rap", "Old school"],
    "Electronic": ["Dance/Electronic", "Electronic dance music", "Electronic","EDM", "House", "Trance", "Techno"],
    "Country":["Country"],
    "Latin":["Latin"]
    # Add more mappings as needed
}


def clean_genre(genre_str):
    # Split the genre string by commas
    genres = genre_str.split(', ')
    # Trim whitespace and convert to a set for unique values
    genres = set(genre.strip().lower() for genre in genres)

    # Find the first matching popular genre
    for popular_genre, synonyms in popular_genres.items():
        if any(genre in genres for genre in map(str.lower, synonyms)):
            return popular_genre
    return genre_str.split(',')[0].strip()


def main(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)
    df['CleanedGenre'] = df['Genre'].apply(clean_genre)

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Update these paths to your actual file locations
    input_file = "New_updated_File_PATH_2_cleaned_genres.csv"
    output_file = "../Final_cleaned_genres_output.csv"
    main(input_file, output_file)


# Load the CSV file
# file_path = 'New_updated_File_PATH_2_cleaned_genres.csv'  # Change this to the actual file path
# data = pd.read_csv(file_path)
#
# # Apply the cleaning function to the "Genre" column
# data['CleanedGenre'] = data['Genre'].apply(clean_genre, popular_genres=popular_genres_set)
#
# # Save the cleaned DataFrame to a new CSV file
# output_file_path = 'Final_cleaned_genres_output.csv'  # Change this accordingly
# data.to_csv(output_file_path, index=False)
# print("Cleaning completed! Cleaned genres saved")
