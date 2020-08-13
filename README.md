# SkepticSystem

This Python script generates trading algorithms by concatenating a list of indicators selected by
a hyperparameter search, iterating the algorithm on a gradient boost, and testing for classifier
accuracy.

The algorithm's goal was to predict whether a trading instrument would post higher or lower in
the next future period.

## Author's Notes

I built this script on the theory that a randomly generated yet boosted algorithm could eventually
be successful, if only for a short period of time. I aimed to generate short-term strategies
constantly which I could swap out once their effectiveness had ceased.

While I was seeing accuracy measures greater than 75%, I was never sure if I accounted sufficiently
for overfitting. I ultimately stopped the project after finding that my code had look-ahead bias.

If I were to attempt algorithmic trading again, I would build on deep reinforcement learning (RL).
Prediction of stock momentum is only one small part of a successful system. Equitable risk
management is paramount. By treating the stock market as a game with rewards, RL could conceivably
train an agent to trade successfully while accounting for risk and market conditions.

## License

See LICENSE for the current license terms.