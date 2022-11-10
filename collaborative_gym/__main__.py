import argparse
from .env_viewer import viewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collaborative Gym Environment Viewer')
    parser.add_argument('--env', default='LiftTask-v0',
                        help='Default Environment: LiftTask-v0')
    args = parser.parse_args()

    viewer(args.env)
