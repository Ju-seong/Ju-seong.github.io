---
layout: post
title: "Self-Nomination for Decentralized CSI Feedback Reduction in MU-MIMO"
date: 2026-04-08
math: true
description: "A short overview of our TWC 2026 paper, Self-Nomination: Deep Learning for Decentralized CSI Feedback Reduction in MU-MIMO Systems."
---

In MU-MIMO systems, UEs typically report CSI regardless of how likely they are to be scheduled. This is natural from a system design perspective, but it can be quite inefficient. As the number of users grows, always-on CSI feedback creates unnecessary uplink overhead, extra UE-side power consumption, and more complexity in BS-side scheduling and precoding. A more direct question is whether each UE can learn, from its own channel, whether its CSI is actually worth feeding back.

## Core idea

Each UE observes its own downlink channel vector $\mathbf{h}_k \in \mathbb{C}^{N}$ and makes a binary feedback decision

$$
f(\mathbf{h}_k; \Theta) \in \{0,1\}.
$$

If $f(\mathbf{h}_k; \Theta)=1$, the UE nominates itself and feeds back CSI. Otherwise, it stays silent. Let $\mathcal{K}$ denote the set of self-nominated UEs. The BS then schedules up to $M$ users from $\mathcal{K}$ and performs precoding. In this work, we mainly consider ZF precoding after UE selection.

![Self-nomination system]({{ "/assets/img/posts/self-nomination/fig1.png" | relative_url }})
*Self-nomination lets each UE decide whether to feed back CSI before BS-side scheduling and precoding.*

We formulate the design under an average sum feedback constraint:

$$
\max_{f(\cdot;\Theta)} \; \mathbb{E}\!\left[\sum_{m\in\mathcal{M}} w_m R_m\right]
\qquad
\text{subject to}
\qquad
\mathbb{E}\!\left[\sum_{k\in\bar{\mathcal{K}}} f(\mathbf{h}_k;\Theta)\right] \le N_{\mathrm{FB}}.
$$

So CSI feedback is no longer treated as something every UE should always send. Instead, it becomes a constrained decision problem.

## Self-nominating DNN

The proposed self-nomination network maps each UE's local CSI to a binary feedback decision. In the full-CSI case, the network sees the full channel vector, so it can use not only channel strength but also spatial structure. This is important in MU-MIMO, where good scheduling depends on more than just strong channels.

A key point is that the decision is made locally at each UE, but the policy is trained centrally using a system-level objective. At inference time, each UE only needs its own CSI.

![Policy-gradient self-nomination architecture]({{ "/assets/img/posts/self-nomination/fig2.png" | relative_url }})
*In the policy-gradient version, self-nomination is modeled as a Bernoulli policy, so training avoids direct differentiation through the hard decision and is easier to adapt across different scheduling rules.*

## Training

We study two training strategies.

The first is a direct optimization approach based on a primal-dual Lagrangian. Since both the binary feedback decision and the BS-side scheduling step are non-differentiable, this approach relies on gradient approximations.

The second is a policy-gradient approach, where self-nomination is modeled as a stochastic policy,

$$
f(\mathbf{h}_k;\Theta) \sim \mathrm{Bernoulli}(p_k),
\qquad
p_k = \sigma(\gamma c_k).
$$

Instead of differentiating through the hard decision, we optimize the expected objective with respect to the policy. This makes the training procedure more modular, since it does not require redesigning scheduler-specific gradient approximations. In our experiments, the policy-gradient-based method is often the stronger option.

## Main result

The main takeaway is that self-nomination can significantly reduce feedback overhead with little or no performance loss.

In the UPA setting, the proposed method outperforms conventional full-feedback baselines because it filters out spatially incompatible users before BS-side scheduling. This is especially important when spatial correlation is strong and simply collecting more CSI does not necessarily help ZF precoding.

![Sum-rate versus number of UEs in the UPA setting]({{ "/assets/img/posts/self-nomination/fig3.png" | relative_url }})
*In the UPA setting, self-nomination achieves strong MU-MIMO sum-rate performance as the number of UEs grows.*

## Feedback reduction

The gain is not only in sum-rate. The proposed method also sharply reduces how many UEs actually send CSI.

In the UPA setting, the number of self-nominated users stays relatively small even as the total number of UEs increases. This reflects the fact that, under strong spatial congestion, only a limited subset of users are spatially suitable for effective MU-MIMO transmission.

![Average number of self-nominated users in the UPA setting]({{ "/assets/img/posts/self-nomination/fig4.png" | relative_url }})
*In the UPA setting, self-nomination keeps the number of feedback users low while maintaining strong sum-rate performance.*

This is useful not only for uplink resource savings, but also for UE energy savings, since non-nominating users can remain silent.

## Beyond simple thresholding

An important question is whether self-nomination is just learning a simple channel-gain threshold. The answer is no.

The fairness result below shows that self-nomination can keep performance close to the full-feedback case while using far fewer feedback transmissions.

![CDF of mean rate under PF scheduling]({{ "/assets/img/posts/self-nomination/fig5.png" | relative_url }})
*Under proportional-fair scheduling, self-nomination maintains fairness close to the full-feedback baseline with much lower feedback cost.*

The next figure gives more insight into the decision rule itself. If the policy were using only a single global threshold, then the minimum channel gain required for nomination would look roughly similar across spatial groups. Instead, larger spatial clusters tend to require higher channel gain for nomination.

![Spatial-cluster-based interpretation of self-nomination]({{ "/assets/img/posts/self-nomination/fig6.png" | relative_url }})
*Users in larger spatial clusters tend to need higher channel gain to be nominated, showing that self-nomination adapts to spatial congestion rather than using a single fixed threshold.*

This means the effective nomination rule depends on both power-domain and spatial-domain information. In that sense, self-nomination behaves more like a spatially adaptive soft threshold than a fixed CQI threshold.

## Takeaway

The main message of this work is that, in MU-MIMO, CSI feedback should be selective rather than universal.

By letting each UE decide whether its CSI is worth feeding back, self-nomination reduces redundant uplink transmissions, simplifies BS-side scheduling, and can improve downlink performance by filtering out spatially incompatible users in advance. It also makes simple random scheduling surprisingly effective. At the UE side, users that do not feed back can remain silent, creating the potential for meaningful power savings.

## Paper

This post is based on the following paper: [J. Park, F. Sohrabi, J. Du, and J. G. Andrews, "Self-Nomination: Deep Learning for Decentralized CSI Feedback Reduction in MU-MIMO Systems," in *IEEE Transactions on Wireless Communications*, vol. 25, pp. 10321-10336, 2026](https://ieeexplore.ieee.org/abstract/document/11351314).