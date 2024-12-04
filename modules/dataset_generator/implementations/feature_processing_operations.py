import pandas as pd
from typing import Optional, List
import logging
from modules.data_structures.processed_dataset import ProcessedDataset
from ..interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)

logging.basicConfig(level=logging.INFO)


class TopNPlayersFeatureProcessor(IFeatureProcessorOperator):
    """
    A feature processor that generates feature vectors for games
    based on the top N players' statistics.
    """

    def __init__(
        self,
        top_n_players: int,
        sorting_criteria: str,
        look_back_window: int,
        player_stats_columns: List[str],
    ):
        self.top_n_players = top_n_players
        self.sorting_criteria = sorting_criteria
        self.look_back_window = look_back_window
        self.player_stats_columns = player_stats_columns

    def get_recent_games(
        self, df: pd.DataFrame, team_id: int, game_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Retrieves the most recent 10 games for a given team before the specified date.
        """
        recent_games = (
            df[(df["TEAM_ID"] == team_id) & (df["GAME_DATE_EST"] < game_date)]
            .sort_values(by="GAME_DATE_EST", ascending=False)
            .head(10)
        )
        return recent_games

    def get_top_players_stats(
        self, df: pd.DataFrame, team_id: int, game_date: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """
        Gets the top N players' stats for a team within the past 10 games.
        """
        recent_games = self.get_recent_games(df, team_id, game_date)
        if recent_games.shape[0] < self.look_back_window:
            return None  # Not enough games to generate stats

        player_stats = recent_games.groupby("PLAYER_ID")[
            self.player_stats_columns
        ].mean()
        top_n_players = player_stats.sort_values(
            by=self.sorting_criteria, ascending=False
        ).head(self.top_n_players)

        # TODO: Figure out if it's possible that less than 8 players come out of this list. That would break the model down the line
        # if top_n_players < 8:
        # find solution: pad the player statistics with zeros to ensure we always get 8 players, or skip the game entirely if the data isn't sufficient?
        # Ensure exactly top_n_players by padding with zeros if necessary
        # if top_n_players.shape[0] < self.top_n_players:
        #     padding = pd.DataFrame(
        #         np.zeros((self.top_n_players - top_n_players.shape[0], len(self.player_stats_columns))),
        #         columns=self.player_stats_columns
        #     )
        #     top_n_players = pd.concat([top_n_players, padding], ignore_index=True)
        # I am unsure about the implications of this on the model

        return top_n_players

    def create_feature_vector(self, top_players: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a feature vector from the top players' statistics, retaining the feature names.

        Example:
        Given the following 'top_players' DataFrame:
        | PLAYER_ID | MIN | PTS | AST | FG_PCT |
        |-----------|-----|-----|-----|--------|
        | 1         | 30  | 15  | 5   | 0.45   |
        | 2         | 28  | 18  | 7   | 0.50   |

        The resulting feature vector will be a DataFrame with columns:
        | home_player_1_MIN | home_player_1_PTS | home_player_1_AST | ... | home_player_2_FG_PCT |
        | 30                | 15                | 5                 | ... | 0.50                 |
        """
        # Create a dictionary for the feature data with unique column names for each player statistic
        feature_data = {}
        for player_index, (_, player_data) in enumerate(top_players.iterrows()):
            for stat_name, value in player_data.items():
                new_column_name = f"home_player_{player_index + 1}_{stat_name}"
                feature_data[new_column_name] = value

        # Create a DataFrame with a single row containing all the top player statistics
        feature_vector_df = pd.DataFrame([feature_data])

        return feature_vector_df

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the input DataFrame to create feature vectors for each game,
        ensuring GAME_ID is preserved for later merging.
        """
        sorted_df = (
            df.drop_duplicates(subset="GAME_ID")
            .assign(GAME_DATE_EST=pd.to_datetime(df["GAME_DATE_EST"]))
            .sort_values(by=["GAME_DATE_EST"])
        )

        feature_vectors = []

        for game_id, game_data in sorted_df.groupby("GAME_ID"):
            team_A_ID, team_B_ID = (
                game_data.iloc[0]["HOME_TEAM_ID"],
                game_data.iloc[0]["VISITOR_TEAM_ID"],
            )
            game_date = game_data.iloc[0]["GAME_DATE_EST"]

            top_n_A = self.get_top_players_stats(df, team_A_ID, game_date)
            top_n_B = self.get_top_players_stats(df, team_B_ID, game_date)

            """
            These are attempts to catch data quality errors and exclude them from final dataset
            """
            if top_n_A is None or top_n_B is None:
                logging.info(
                    f"Skipping game {game_id} due to insufficient player data."
                )  # Expected for initial games close to beginning of dataset
                continue

            if (
                top_n_A.shape[0] != self.top_n_players
                or top_n_B.shape[0] != self.top_n_players
            ):
                logging.warning(
                    f"Warning: Unexpected number of players for game {game_id}. Skipping this game."
                )
                continue

            if (game_data["PTS_home"] < 0).any() or (game_data["PTS_away"] < 0).any():
                logging.error(
                    f"Invalid score detected for game {game_id}. Skipping this game."
                )
                continue

            if game_data[["PTS_home", "PTS_away"]].isna().any().any():
                logging.warning(
                    f"Missing score detected for game {game_id}. Skipping this game."
                )
                continue

            feature_vector_A = self.create_feature_vector(top_n_A)
            feature_vector_B = self.create_feature_vector(top_n_B)

            feature_vector = pd.concat([feature_vector_A, feature_vector_B], axis=1)
            feature_vector["GAME_ID"] = game_id
            feature_vector["final_score_A"] = game_data.iloc[0]["PTS_home"]
            feature_vector["final_score_B"] = game_data.iloc[0]["PTS_away"]

            feature_vectors.append(feature_vector)

        return pd.concat(feature_vectors, ignore_index=True)

    def extract_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the labels (final scores) from the input DataFrame.
        Removes rows with missing scores.
        """
        labels = df[["GAME_ID", "PTS_home", "PTS_away"]].dropna().drop_duplicates()
        return labels

    def process(self, dataframe: pd.DataFrame) -> ProcessedDataset:
        """
        Processes the input DataFrame to generate a ProcessedDataset with features and labels.

        Returns:
            ProcessedDataset: The processed dataset with features and labels.
        """
        features_df = self.process_features(dataframe).set_index("GAME_ID")
        labels_df = self.extract_labels(dataframe).set_index("GAME_ID")

        # Ensure that only matching GAME_IDs are kept (we dropped duplicates on both sides and took out rows with data quality issues)
        merged_df = features_df.join(labels_df, how="inner")

        # Separate features and labels
        final_features = merged_df.drop(columns=["PTS_home", "PTS_away"])
        final_labels = merged_df[["PTS_home", "PTS_away"]]

        return ProcessedDataset(features=final_features, labels=final_labels)
