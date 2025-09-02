# vector_db_client.py
import os
from google.cloud import aiplatform

class VertexAIVectorClient:
    def __init__(self, project_id: str, location: str, endpoint_id: str, deployed_index_id: str):
        aiplatform.init(project=project_id, location=location)
        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint_name=endpoint_id)
        self.deployed_index_id = deployed_index_id
        
    def add_vectors(self, datapoints):
        """Adds new vectors to the index."""
        self.index_endpoint.add_vectors(datapoints)
        
    def find_neighbors(self, query_vector: list, num_neighbors: int):
        """Queries the index for nearest neighbors."""
        return self.index_endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_vector],
            num_neighbors=num_neighbors
        )