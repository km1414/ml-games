import argparse
from gamer import Gamer
from models.random_model import RandomModel
from models.policy_gradients_model import PolicyGradientsModel



def main():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='ml-games')

    # Add arguments
    parser.add_argument('-game', type=str, help='Select game name.')
    parser.add_argument('-model', type=str, help='Select model name.')
    parser.add_argument('-n_games', type=int, default=10000, help='Number of games to play.')

    # Parse arguments
    args = parser.parse_args()

    # Initialize mandatory objects
    gamer = Gamer(args.game)
    model = None
    if args.model == 'RandomModel':
        model = RandomModel()
    if args.model == 'PolicyGradientsModel':
        model = PolicyGradientsModel()
    gamer.connect_model(model)

    print('Game:', args.game)
    print('Model:', args.model)
    print('Number of games:', args.n_games)

    # Play
    for i in range(args.n_games):
        gamer.play_game()


if __name__ == '__main__':
    main()