import pandas as pd
from typing import Tuple

class TopNPlayersFeatureProcessor:
    """
    Processes features by selecting the top N players based on given criteria.
    """

    def process(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Placeholder implementation for processing features and labels
        features_df = dataframe.drop(columns=['label'])
        labels_df = dataframe[['label']]
        return features_df, labels_df


### Swap for below after fixing it

# import pandas as pd


# class FeatureProcessorParams:
#     """
#     A class that stores parameters and provides methods for processing player statistics
#     and generating feature vectors for games.
#     """

#     def __init__(self, top_n_players: int = 8, sorting_criteria: str = "MIN"):
#         self.top_n_players = top_n_players
#         self.sorting_criteria = sorting_criteria
#         self.player_stats_columns = [
#             "MIN",
#             "PTS",
#             "AST",
#             "TO",
#             "PLUS_MINUS",
#             "OREB",
#             "DREB",
#             "PF",
#             "FG3_PCT",
#             "FG_PCT",
#             "FT_PCT",
#         ]

#     def get_top_players_stats(self, df, team_id, game_date):
#         """
#         Gets the top N player stats, based on sort_by, for a team within the past 10 games from a given game_date.
#         """
#         # Filter df for games where team_id played, and that came before game_date.
#         # Then reduce that subset to the latest 10 games by sorting by game date and taking the latest 10 games.
#         recent_games = (
#             df[(df["TEAM_ID"] == team_id) & (df["GAME_DATE_EST"] < game_date)]
#             .sort_values(by="GAME_DATE_EST", ascending=False)
#             .head(10)
#         )

#         # Abort if there are fewer than 10 recent games
#         if recent_games.shape[0] < 10:
#             return None

#         player_stats = recent_games.groupby("PLAYER_ID")[
#             self.player_stats_columns
#         ].mean()
#         top_players = player_stats.sort_values(
#             by=self.sorting_criteria, ascending=False
#         ).head(
#             self.top_n_players
#         )  # TODO: Not sure if this could even happen, but we could consider the possibility of ending up with less than 8 players and decide how to handle that
#         return top_players

#     def process_features(self, df):
#         """
#         Processes the input DataFrame to create feature vectors for each game.
#         """
#         df["GAME_DATE_EST"] = pd.to_datetime(
#             df["GAME_DATE_EST"]
#         )  # Ensuring the date column is in datetime format
#         df.sort_values(by=["GAME_DATE_EST"], inplace=True)  # Sorting by date

#         final_features = []

#         grouped = df.groupby("GAME_ID")

#         for (
#             game_id,
#             game_data,
#         ) in (
#             grouped
#         ):  # loop through the groupings in ascending order. The grouped operation respects the previous sorting of games
#             # Extracting final scores and team IDs
#             final_score_A = game_data.iloc[0]["PTS_home"]
#             final_score_B = game_data.iloc[0]["PTS_away"]

#             team_A_ID = game_data.iloc[0]["HOME_TEAM_ID"]
#             team_B_ID = game_data.iloc[0]["VISITOR_TEAM_ID"]

#             game_date = game_data.iloc[0]["GAME_DATE_EST"]

#             # Getting top 8 player stats for both teams
#             top_eight_A = self.get_top_players_stats(df, team_A_ID, game_date)
#             top_eight_B = self.get_top_players_stats(df, team_B_ID, game_date)

#             # Skip if either team does not have enough recent games
#             if top_eight_A is None or top_eight_B is None:
#                 continue

#             # Create feature vector
#             feature_vector = []

#             # Add home team top players stats
#             for player_id, stats in top_eight_A.iterrows():
#                 feature_vector.extend(stats.values)

#             # Add visitor team top players stats
#             for player_id, stats in top_eight_B.iterrows():
#                 feature_vector.extend(stats.values)

#             # Add final points for home and visitor teams
#             feature_vector.append(final_score_A)
#             feature_vector.append(final_score_B)

#             final_features.append(feature_vector)

#         # Create a DataFrame from the feature vectors
#         columns = (
#             [
#                 f"home_player_{i}_{stat}"
#                 for i in range(1, 9)
#                 for stat in self.player_stats_columns
#             ]
#             + [
#                 f"visitor_player_{i}_{stat}"
#                 for i in range(1, 9)
#                 for stat in self.player_stats_columns
#             ]
#             + ["final_score_A", "final_score_B"]
#         )

#         final_df = pd.DataFrame(final_features, columns=columns)

#         return final_df