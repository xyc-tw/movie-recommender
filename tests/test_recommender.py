from recommender import recommend_random

def test_recommend_random():
    assert len(recommend_random(3)) == 3