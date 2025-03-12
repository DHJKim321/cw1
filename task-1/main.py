import argparse
from test import test_knn_performance

def main():
    """
    Main function to run the Top-K Nearest Neighbors performance test.
    Allows specifying a test file and distance metric via command-line arguments.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Top-K Nearest Neighbors Performance Test")
    parser.add_argument(
        "--test_file", type=str, default="", 
        help="Path to JSON test file or leave empty for random data"
    )
    parser.add_argument(
        "--distance_metric", type=str, choices=["l2", "cosine", "dot", "manhattan"], default="l2",
        help="Distance metric to use (l2, cosine, dot, or manhattan)"
    )
    
    args = parser.parse_args()

    # Run the test with provided parameters
    test_knn_performance(test_file=args.test_file, dis_metric=args.distance_metric)

if __name__ == "__main__":
    main()
