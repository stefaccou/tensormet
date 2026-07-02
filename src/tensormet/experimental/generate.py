from __future__ import annotations

import numpy as np



class TuckerGenMixin:
    def generate(self, start_sent, els_to_add=6, temperature=0.9, k=20):
        # The model order is the number of roles; we condition on the leading
        # (order - 1) roles and predict the final role at each step.
        ctx_len = len(self.roles) - 1
        target_role = self.roles[-1]

        if len(start_sent) < ctx_len:
            raise ValueError(
                f"start_sent needs at least {ctx_len} tokens for an "
                f"order-{len(self.roles)} model, got {len(start_sent)}."
            )

        full_tuple = tuple(start_sent[:ctx_len])
        for i in range(els_to_add):
            # Slide a window of the trailing ctx_len tokens; leave the final slot open.
            goal_sent = full_tuple[i:i + ctx_len] + ("new",)
            predictions = self.get_expected_element(
                goal_sent, target_role, method="excluded", verbose=False, k=k
            )
            tokens = [p['token'] for p in predictions]
            scores = np.array([p['score'] for p in predictions])
            scaled_scores = scores / temperature
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
            probabilities = exp_scores / np.sum(exp_scores)
            sampled_token = np.random.choice(tokens, p=probabilities)
            full_tuple += (str(sampled_token),)

        return full_tuple