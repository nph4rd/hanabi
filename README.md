# hanabi

### Overview
- **Environment ID**: `hanabi`
- **Short description**: Cooperative Hanabi card game where players work together to build fireworks
- **Tags**: multi-agent, multi-turn, cooperative
- **Repo**: https://github.com/nph4rd/hanabi

### Task
- **Type**: multi-turn
- **Parser**: XMLParser (fields: `action`)
- **Rubric**: Score-based reward (0-25 points)

Players take turns performing one of three actions:
- **Play** (`P0`-`P4`): Play a card from your hand
- **Discard** (`D0`-`D4`): Discard a card to gain an info token
- **Hint** (`{player}H{color/rank}`): Give a hint to another player (e.g., `1HR` or `2H3`)

The game ends when all fireworks are completed (25 points), all lives are lost, or the deck runs out.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval hanabi
```

Configure model and sampling:

```bash
uv run vf-eval hanabi -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"num_players": 3}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `2000` | Number of training examples (each with a unique seed) |
| `num_eval_examples` | int | `20` | Number of evaluation examples |
| `num_players` | int | `2` | Number of players (must be > 1; hand size is 5 for 2-3 players, 4 for more) |
| `max_turns` | int | `-1` | Maximum turns per game (-1 for unlimited) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Final game score (0-25, sum of completed firework ranks) |

### Training

- Tested against [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) at [a1713c8](https://github.com/PrimeIntellect-ai/prime-rl/tree/a1713c81c4ec7a86b46399aa7f396675345ba0ed).
- Data for SFT can be found in [here](https://huggingface.co/datasets/nph4rd/hanabi).
- Synth example config [here](https://github.com/nph4rd/hanabi-back/blob/main/synth.toml).
- SFT example config [here](https://github.com/nph4rd/hanabi/blob/main/sft.toml).
- RL example config [here](https://github.com/nph4rd/hanabi/blob/main/rl.toml).
