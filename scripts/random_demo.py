from shotgun_king.core import ShotgunKingLite


def main():

    game = ShotgunKingLite()

    state = game.reset(seed=42)

    print(game.render())

    while not (
        state.won
        or state.lost
        or state.timeout
    ):

        actions = game.legal_actions()

        action = game.rng.choice(actions)

        print("\naction =", action)

        state, events = game.step(
            int(action)
        )

        print("events =", events)

        print(game.render())

    print("\nEpisode finished.")


if __name__ == "__main__":
    main()