import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

# ---------- Configuration ----------
# This key can be obtained from dev.spotify.com
SPOTIPY_CLIENT_ID = '<Get client key>'
SPOTIPY_CLIENT_SECRET = '<Your secret key>'
SPOTIPY_REDIRECT_URI = 'http://localhost:8080'  # Redirect URI specified in your Spotify Developer Dashboard
SCOPE = 'playlist-modify-public'  # Permissions needed
PLAYLIST_NAME = 'BillBoard-train'
FILENAME = 'completeCleanedLabeled.csv'

# ---------- Authentication ----------
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=SCOPE))

df = pd.read_csv(FILENAME)

user_id = sp.current_user()['id']
num_songs = len(df)
songs_per_playlist = num_songs // 4
playlists = [df.iloc[i:i + songs_per_playlist] for i in range(0, num_songs, songs_per_playlist)]


if len(playlists) > 4:
    playlists[3] = pd.concat([playlists[3], playlists[4]])
    playlists = playlists[:4]

for i, playlist_df in enumerate(playlists):
    playlist_name = f"BillBoard Part {i + 1}"
    playlist = sp.user_playlist_create(user=sp.current_user()['id'], name=playlist_name, public=True,
                                       description=f'Generated Playlist Part {i + 1}')
    playlist_id = playlist['id']

    position = 0
    track_ids = []
    for index, row in playlist_df.iterrows():
        print(f"Adding Song {index + 1} to {playlist_name}: {row['SongTitle']}")
        query = f"{row['SongTitle']} {row['Artist']}"
        results = sp.search(q=query, type='track', limit=1)
        tracks = results['tracks']['items']
        if tracks:
            track_id = tracks[0]['id']
            sp.playlist_add_items(playlist_id=playlist_id, items=[track_id], position=position)
            position += 1


print(f"Playlist '{PLAYLIST_NAME}' created successfully!")