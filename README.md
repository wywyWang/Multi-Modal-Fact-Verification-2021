# Official code of Team Yao at Multi-Modal-Fact-Verification-2022
A Multi-Modal Fact Verification dataset released as part of the [De-Factify workshop in AAAI-22](https://aiisc.ai/defactify/).
We won the **fifth place** and the technical report will be published in the near future.

## Challenge
The task is to find out support, insufficient-evidence and refute between given claims.

## Dataset
- Train set: 35,000, 7,000 for each class.
- Validation set: 7,500, 1,500 for each class.
- Test set: 7,500, 1,500 for each class.
For more details, please refer to [FACTIFY: A Multi-Modal Fact Verification Dataset](https://www.researchgate.net/publication/356342935_FACTIFY_A_Multi-Modal_Fact_Verification_Dataset).

## Metric
F1 averaged across the 5 categories. The final ranking would be based on the weighted average F1 score.

## Method Overview
<div align="left">
<img src="confusion_matrix.png"/>
</div>
## Leaderboard
<div align="left">
<img src="Leaderboard.png"/>
</div>