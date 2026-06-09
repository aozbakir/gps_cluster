# Unsupervised Kinematic Clustering of GNSS Velocities Reveals Hidden Seismic Hazard in Slowly Deforming Fault Systems

Ali Deger Özbakır

---

## Target journal: Nature Communications

**Core argument:** Slow-deforming fault systems and geometrically complex fault junctions are the least well-characterised sources of seismic hazard. Block models require fault boundaries to be prescribed a priori and cannot identify boundaries where none are mapped. We show that Euler-vector clustering of raw GNSS velocities — an unsupervised approach making no prior assumption about fault geometry — recovers kinematically distinct blocks at two fault junctions that produced major unexpected multi-fault rupture sequences: the Kahramanmaraş 2023 (Mw 7.8 + 7.7) and [Marlborough 2016 Mw 7.8 — to be analysed]. In both cases the recovered block boundaries correspond to fault strands that were either unmapped, underweighted in hazard assessments, or invisible to previous GPS clustering methods. Slip rates computed directly from the Euler pole pairs agree with geological estimates where available and provide new constraints where they do not.

**Key additions needed before submission:**
- Marlborough / Kaikōura (NZ) case study
- Slip rate comparison table (geodetic vs geological for EAF, Sürgü-Çardak, Isparta Angle bounding faults, and Marlborough faults)
- Coulomb stress or seismic moment argument connecting block kinematics to rupture

---

---

## Abstract

The Anatolia–Aegean domain represents a distributed deformation zone sub-divided into large plate-like units, and its dense, decades-long GNSS network makes it one of the best-instrumented tectonic regions in the world. Two end-member approaches describe continental deformation here. The continuum strain-rate approach makes no assumptions about fault geometry, but full resolution of velocity gradients requires spatial sampling of about 0.25 of the locking depth, which is typically lower than the available GPS station spacing (Haines et al., 2015). Block models are tractable with current data and have been successful in calculating slip rates on major active faults — a major input for seismic hazard assessment — but idealise deformation as an assembly of internally rigid blocks whose boundaries must be prescribed from mapped faults and seismicity; slip-rate estimates consequently vary significantly across studies. Furthermore, a common pre-analysis step removes the best-fitting Euler pole of the dominant block to expose residual deformation and guide boundary identification; in a slowly but genuinely deforming region, however, the degree of internal coherence is itself an open question the velocity field should constrain, not a premise imposed upon it. Clustering of GPS velocities has been proposed as an objective alternative to prescribed block boundaries (Simpson et al., 2012; Savage, 2018), and has been applied to Turkey using velocity-space algorithms in Eurasia-fixed frames (Özdemir & Karslıoğlu, 2019; Kılıç & Özarpacı, 2022). These studies, however, cluster two-dimensional velocity vectors without inverting for rotation poles, and work in a reference frame that pre-removes the dominant rotation — leaving both sources of prior assumption intact. I implement and extend the iterative Euler-vector clustering of Savage (2018), adding a multiscale initialisation scheme and a sequential F-test on reduced χ² for model selection, and apply it directly to raw GNSS velocities so that block boundaries and rotation poles emerge simultaneously from the data. [Results placeholder.]

---

## 6.1 Motivation

[To be written — see session notes.]

---

## 6.3 Previous Work

### 6.3.1 Comparison of Clustering Studies for Turkey

All three clustering studies applied to the Turkey–Aegean domain converge on k = 5 as the statistically preferred number of kinematic blocks. This convergence is methodologically meaningful — the three studies use different algorithms and model-selection criteria — but it should be noted that Kılıç & Özarpacı (2022) use the same CORS-TR velocity field as Özdemir & Karslıoğlu (2019), so the agreement on k between those two reflects independent methods applied to identical data, not independent observations. The present study uses a larger and independently processed velocity field (836 stations, ITRF14), making the three-way agreement on k = 5 a genuinely convergent result. Nevertheless, the spatial definitions of those five units differ in important ways that reflect the methodological choices of each study.

**Özdemir & Karslıoğlu (2019)** present a homogeneous velocity field from ~210 CORS-TR permanent stations and apply k-means, HAC, and Gaussian mixture model (GMM) clustering in a Eurasia-fixed reference frame. Their preferred result — soft GMM clustering at k = 5 with full, shared covariance matrices — recovers five clusters: the Eurasian block along the Black Sea coast, the Arabian block in the southeast, and three Anatolian sub-units (East Anatolian, West Anatolian, and SouthWest Anatolian blocks). The Eurasian and Arabian clusters emerge immediately and with high posterior probability, as their velocity vectors are tightly grouped and well-separated from the Anatolian interior in the Eurasia-fixed frame. The three Anatolian units are more loosely defined: maximum posterior probabilities fall below 0.9 for stations near the NAF and in western Anatolia, and block boundaries shift by up to 100 km depending on the covariance structure chosen for the GMM. The authors compare their boundaries against those of Nyst & Thatcher (2004), Reilinger et al. (2006), and Aktuğ et al. (2009), finding broad agreement on the major divisions but disagreement on the extent of the southwest Anatolian and Aegean units — a sensitivity they attribute to the sparsity of the CORS-TR network in western Turkey. Crucially, all clustering is performed on Eurasia-fixed velocities: the dominant rotational signal of Anatolia is removed before any partition is attempted, and Euler poles are computed after the fact as a descriptive statistic rather than driving the cluster assignments.

**Kılıç & Özarpacı (2022)** apply five individual clustering algorithms (BIRCH, k-means, mini-batch k-means, HAC, spectral clustering) to the same CORS-TR velocity field, followed by three ensemble consensus methods (HGBF, MCLA, NMF). Their motivation is to overcome the algorithm-dependence of single-method clustering: stations that change assignment between individual runs are resolved by majority vote. Gap statistic selects k = 5, matching Özdemir & Karslıoğlu. The recommended MCLA ensemble yields a partition similar in structure to the earlier study — Eurasia, Arabia, and three Anatolian sub-units — but with tighter, less ambiguous boundaries in the central and western regions where individual algorithms disagree. Compared to Özdemir & Karslıoğlu, the MCLA result shows a more decisive Aegean delineation and a cleaner NAF as the northern boundary of the Anatolian units. However, the two prior studies diverge in the eastern part of Turkey: Kılıç & Özarpacı find that their cluster boundaries depart from the Reilinger et al. (2006) block model near the EAF, which they attribute to velocity gradients not aligning with mapped fault traces — whereas Özdemir & Karslıoğlu's boundaries in the same region follow the published block geometry more closely, likely because their GMM covariance structure favours compact, convex clusters that conform to known fault-bounded units. Both studies inherit the same fundamental limitation: the ensemble operates on Eurasia-fixed 2D velocity vectors, so the reference frame choice and the absence of Euler-pole physics are shared by all constituent algorithms.

**This study** applies Euler-vector clustering (Savage, 2018) to 836 GNSS stations processed uniformly in ITRF14, without removing any reference Euler pole prior to analysis. The algorithm iteratively inverts for one Euler pole per cluster and reassigns stations to minimise velocity residuals with respect to those poles, so the spatial partition and the rotation poles are determined simultaneously rather than sequentially. Model selection by sequential F-test on reduced χ² selects k = 5, consistent with both prior studies. Because raw ITRF velocities are used, the strong rotational signal of Anatolia is not suppressed before clustering: the algorithm must account for it through the Euler pole inversion, and the resulting boundaries reflect kinematic groupings rather than similarity in residual velocity magnitude. The eastern partition — where both prior studies show diffuse or algorithm-sensitive boundaries — is more decisively resolved here because the Euler-pole fit criterion assigns each station to whichever rotation best predicts its absolute motion, irrespective of the station's position relative to mapped faults. Three specific features of the solution that differ from the prior studies deserve detailed discussion.

*NAF stations assigned to Cluster 5.* Stations along the northern NAF corridor (40.5°–41.5°N, 27°–35°E) are predominantly assigned to Cluster 5, whose Euler pole at 65.9°N, 7.9°W with a rate of 0.44°/Ma is consistent with the Arabian plate's absolute rotation in ITRF. Geographically these stations lie in eastern Turkey and the Caucasus foreland, not on the Arabian plate itself; the identification rests on the pole position and rotation rate alone. Their assignment to this cluster is counter-intuitive, since they sit on the Anatolian side of the NAF and have no direct kinematic connection to Arabia. The most likely explanation is elastic interseismic coupling across the locked NAF: stations near the fault are velocity-deficient relative to their block interior, shifting their absolute ITRF velocities toward those of a more slowly rotating plate, and the algorithm assigns them to Cluster 5 because that Euler pole minimises their residual. A geometric test confirms that this is not a true kinematic relationship: the angular distance from the Cluster 5 pole to successive NAF waypoints varies by ±2°, far from the ±0.5° consistency that would be expected if the NAF were a small circle around that pole. This is a known limitation of Euler-vector clustering near major locked faults, and motivates a future extension in which an elastic interseismic correction is applied to the velocity field before partitioning.

*Resolution of the southern NAF branch.* The boundary between Clusters 1 and 3 in the longitude band 29°–35°E aligns with the southern branch of the NAF rather than the northern Marmara strand. Cluster 1 (centred at 39.2°N, 31.1°E) captures the block south of the southern NAF, including stations in the Mudurnu–Bolu corridor (39.0°–40.2°N), while the northern strand at 40.5°–41.5°N is split between Clusters 3, 4, and 5. This two-strand resolution is absent in both prior studies, which treat the NAF as a single boundary in the Eurasia-fixed frame. It suggests that the Euler-pole fit is sensitive to the kinematic contrast across the southern branch, which accommodates a non-negligible fraction of the total Anatolia–Eurasia relative motion in the western segment. The resolution of this secondary strand without prescribing it as a block boundary is a direct consequence of operating on raw velocities: in the Eurasia-fixed frame the velocity contrast across the southern NAF is small enough to be smoothed over by the clustering objective.

*Cluster 2 and the Isparta Angle.* Cluster 2 (N = 110, Euler pole 38.9°N, 29.1°E, 2.52°/Ma) occupies southwestern Turkey, centred at 37.6°N, 29.0°E with an eastern boundary near 32°–34°E. This spatial extent closely follows the Isparta Angle — the curved fold-thrust belt where the Lycian nappe front (NE-SW trending) meets the Tauride fold belt (NW-SE trending), forming a triangular indenting wedge bounded by the Sultandağı and Burdur faults to the north and southwest respectively. At 2.52°/Ma, Cluster 2 has the highest rotation rate of all five clusters, and its Euler pole falls within the cluster's own geographic footprint — together these indicate a compact, rapidly rotating block, consistent with the Isparta Angle behaving as a rigid indenter caught between Aegean extension to the west and northward Arabian convergence to the south. Neither Özdemir & Karslıoğlu (2019) nor Kılıç & Özarpacı (2022) isolate this unit: in both studies the region is absorbed into a broader West or SouthWest Anatolian cluster, because the high rotation rate that distinguishes it from surrounding Anatolia is masked when velocities are expressed in the Eurasia-fixed frame before clustering.

The three studies are compared in Figure [X]. Despite convergence on k = 5, the underlying methodological differences — reference frame, clustering objective, and whether rotational physics drives the partition or merely describes it post hoc — produce measurably different boundary positions, particularly in western Anatolia, around the Isparta Angle, and along the NAF transition zones. The present study is the only one that yields Euler poles as a primary output, making the results directly comparable to block model solutions without additional processing steps.

---

## NAF Slip Rates from Full Turkey k=5 Clustering

Slip rates computed between adjacent cluster pairs along the NAF using `fault_slip_rate()`. Strike = 270° (pure E-W). Sign convention: +strike-slip = right-lateral, +fault-normal = opening (extension).

### NAF northern strand — C3 (central Anatolia) relative to C5 (Eurasian/NAF-belt)

| Location | Total (mm/yr) | Strike-slip (mm/yr) | Fault-normal (mm/yr) |
|---|---|---|---|
| Ganos–Saros (40.75°N, 27.3°E) | 19.4 | +18.5 (RL) | +5.8 (conv.) |
| Kumburgaz / Marmara (40.80°N, 28.5°E) | 19.1 | +18.6 | +4.4 |
| Adalar / Marmara (40.78°N, 29.1°E) | 19.0 | +18.6 | +3.7 |
| Izmit 1999 Mw 7.6 (40.72°N, 30.2°E) | 18.7 | +18.6 | +2.4 |
| Düzce 1999 Mw 7.2 (40.77°N, 31.1°E) | 18.7 | +18.7 | +1.4 |
| 1944 rupture / Bolu (40.80°N, 32.2°E) | 18.7 | +18.7 | +0.1 |
| 1943 rupture / Kastamonu (40.58°N, 33.0°E) | 18.4 | +18.4 | −0.8 |
| 1943 rupture / Tosya (41.00°N, 34.9°E) | 19.1 | +18.9 | −3.0 |
| 1942 rupture (40.70°N, 36.6°E) | 19.1 | +18.5 | −5.0 |
| 1939 rupture / Erzincan (40.30°N, 37.8°E) | 19.0 | +17.9 | −6.3 |

Right-lateral rate is remarkably uniform along strike (18.4–18.9 mm/yr). Fault-normal component transitions from convergence in the west (+5.8 mm/yr at Ganos) to extension in the east (−6.3 mm/yr at Erzincan) — consistent with the restraining-to-releasing bend geometry of the NAF. Published geodetic estimates: Reilinger et al. (2006) ~20–25 mm/yr; Ergintav et al. (2023) ~18–22 mm/yr in western Marmara. Our 18.5–18.9 mm/yr is slightly low, likely because C5 contains NAF-proximal stations with elastic velocity deficits rather than pure Eurasian velocities.

### NAF northern strand — C1 (western Anatolia) relative to C5

| Location | Total (mm/yr) | Strike-slip (mm/yr) | Fault-normal (mm/yr) |
|---|---|---|---|
| Ganos–Saros (40.75°N, 27.3°E) | 24.6 | +23.2 | +8.1 |
| Kumburgaz (40.80°N, 28.5°E) | 24.2 | +23.4 | +6.5 |
| Adalar (40.78°N, 29.1°E) | 24.1 | +23.4 | +5.6 |
| Izmit (40.72°N, 30.2°E) | 23.7 | +23.4 | +4.1 |
| Düzce (40.77°N, 31.1°E) | 23.6 | +23.5 | +2.8 |

23–23.5 mm/yr right-lateral — closer to the total Anatolia–Eurasia motion budget. The higher rate reflects C1's faster-moving western Anatolian stations.

### NAF southern branch — C1 (western Anatolia) relative to C5

| Location | Total (mm/yr) | Strike-slip (mm/yr) | Fault-normal (mm/yr) |
|---|---|---|---|
| Yenice–Gönen (40.0°N, 27.5°E) | 23.5 | +22.1 | +7.8 |
| Mustafakemalpaşa (40.0°N, 28.5°E) | 23.1 | +22.2 | +6.5 |
| Ulubat (40.2°N, 29.0°E) | 23.2 | +22.5 | +5.8 |
| Iznik-Mekece (40.4°N, 29.8°E) | 23.3 | +22.9 | +4.7 |
| Geyve (40.5°N, 30.4°E) | 23.4 | +23.0 | +3.8 |

22–23 mm/yr right-lateral on the southern branch — comparable to the northern strand at equivalent longitudes. Implies significant strain partitioning between the two branches in the 27–30°E segment, consistent with the cluster boundary aligned with the southern strand rather than the Marmara northern strand.

### Notes for paper

- The C3/C5 slip rate (18.5 mm/yr) likely underestimates the true rate because C5 contains elastically loaded NAF-proximal stations. An elastic correction (interseismic locking model) applied before clustering would sharpen this.
- The C1/C5 rate (23–24 mm/yr) better represents the total Anatolia–Eurasia motion, agreeing with Reilinger et al. (2006).
- The fault-normal gradient (convergence → extension west to east) is a new geodetic constraint on the along-strike variation in NAF kinematics.
- No prior clustering study has computed fault slip rates as a primary output.

---

## EAF Case Study — East Anatolian Fault Region (34–42°E, 36–39°N)

### Data and setup

155 stations from the full 836-station ITRF14 velocity field, filtered to the EAF box. Euler-vector clustering applied with multiscale initialisation (50 restarts) and no reference frame rotation removed prior to analysis. Gap statistic and sequential F-test used for model selection independently.

### Model selection

Gap statistic selects k = 2. F-test on reduced χ² selects k = 7. The divergence between the two criteria is informative: the EAF creates such a dominant velocity contrast that the velocity-space null distribution identifies k = 2 immediately, while the F-test detects that much kinematic structure remains in the residuals (χ²_red = 800 at k = 2, dropping to 515 at k = 3, 320 at k = 4).

### k = 2

Two clusters separated cleanly along the EAF:
- C1 (N = 70, pole 50.2°N, 0.5°W, 0.54°/Ma): Arabian plate, centred at 37.4°N, 39.0°E
- C2 (N = 85, pole 43.2°N, 22.3°E, 0.97°/Ma): Anatolian block, centred at 38.0°N, 36.0°E

The Mw 7.8 (6 Feb 2023) epicentre falls on the boundary between C1 and C2. χ²_red = 800.

### k = 3 — key result

A third cluster (C3, N = 14) emerges centred at 37.5°N, 37.6°E — precisely the Kahramanmaraş junction where the EAF meets the Sürgü-Çardak fault. Its Euler pole (50.3°N, 12°W, 0.40°/Ma) is distinct from both Arabia and Anatolia and has a slower rotation rate. The Mw 7.7 epicentre (Sürgü-Çardak rupture, second event) falls within this isolated cluster. Adding 14 stations to a third cluster reduces χ²_red by 35% (800 → 515), indicating a kinematically coherent unit rather than a statistical artefact.

C1 and C2 poles are stable from k = 2 (pole positions shift by < 1°).

### k = 4

The Kahramanmaraş wedge cluster tightens to N = 12 at 37.5°N, 37.7°E with pole 50.3°N, 12.3°W — confirming its distinctness is not an artefact of under-clustering. The Anatolian block splits into a western unit (N = 42, centre 38.2°N, 35.5°E) and a central unit (N = 40, centre 37.8°N, 36.5°E). χ²_red = 320.

### Scale dependence — methodological note

The Kahramanmaraş wedge is invisible in the full Turkey analysis at k = 5. The dominant Anatolian rotational signal overwhelms the subtle kinematic contrast at the fault junction at the national scale. The EAF regional analysis constitutes a second-level decomposition: the full-Turkey clustering identifies the eastern Anatolian domain as one of five first-order units; the focused regional analysis then resolves sub-block structure within that unit without prescribing any internal boundaries. The prior knowledge is in the choice of analysis domain (the EAF box), not in the location of the recovered boundary.

### Slip rates from adjacent block Euler poles

**EAF — Arabia (C_east) relative to Anatolia (C_west), strike 240°:**

| Point | Total (mm/yr) | Strike-slip (mm/yr) | Fault-normal (mm/yr) |
|---|---|---|---|
| Amanos (37.2°N, 36.1°E) | 7.5 | −7.4 (left-lateral) | −1.3 (convergence) |
| Pazarcık / 2023 epicentre (37.5°N, 36.8°E) | 7.4 | −7.4 | −0.6 |
| Erkenek (37.8°N, 37.5°E) | 7.4 | −7.4 | +0.1 |
| Pütürge (38.1°N, 38.2°E) | 7.4 | −7.4 | +0.9 |
| Palu (38.5°N, 39.0°E) | 7.6 | −7.4 | +1.7 (extension) |

Left-lateral rate of 7.4 mm/yr along the full EAF. Compares well with published geodetic estimates (Reilinger et al. 2006: ~9 mm/yr; McClusky et al. 2000: ~6–9 mm/yr) and geological estimates (6–10 mm/yr). Slight convergence in the south (Amanos) grading to slight extension in the north (Palu) — consistent with the restraining/releasing bend geometry of the EAF.

**Sürgü-Çardak — KMŞ wedge (C_mid) relative to Arabia (C_east), strike 120°:**

| Point | Total (mm/yr) | Strike-slip (mm/yr) | Fault-normal (mm/yr) |
|---|---|---|---|
| (37.7°N, 36.5°E) | 3.8 | +1.0 (right-lateral) | +3.7 |
| (37.5°N, 37.0°E) | 3.9 | +1.0 | +3.8 |
| (37.3°N, 37.5°E) | 4.0 | +1.0 | +3.8 |

~4 mm/yr total, predominantly fault-normal, with modest right-lateral component (~1 mm/yr). The dominant fault-normal component warrants further investigation: it may indicate that the wedge is being extruded northward between the EAF and Sürgü-Çardak rather than sliding laterally. The Euler pole for the wedge cluster is based on only 14 stations and the uncertainty is large — slip rate estimates here should be treated as first-order constraints pending a denser network. No published geodetic slip rate exists for the Sürgü-Çardak fault to compare against; geological estimates give ~1–3 mm/yr right-lateral (Kaymakçı et al. 2006 [check]).

### What prior studies could not resolve

Neither Özdemir & Karslıoğlu (2019) nor Kılıç & Özarpacı (2022) performed a focused regional analysis of the EAF zone. Both treat the entire EAF corridor as a single block or transition zone in their Eurasia-fixed five-cluster solutions. The velocity-space clustering objective, operating on residual velocities from which the dominant Anatolian rotation has been removed, suppresses the rotational contrast that distinguishes the Kahramanmaraş wedge from the surrounding blocks.

---

## Marlborough Fault System — Case Study Results

**Setting:** The Marlborough Fault System (MFS), South Island, New Zealand. The 2016 Kaikōura earthquake (Mw 7.8) ruptured at least 12 fault segments simultaneously — a rupture pattern considered extremely unlikely under standard hazard models because the faults were mapped as separate structures with no prescribed kinematic coupling. The MFS accommodates the transition from the Alpine Fault (pure strike-slip) to the Hikurangi subduction zone (oblique convergence) through a 150 km-wide array of NE-trending right-lateral faults: Hope, Clarence, Awatere, and Wairau.

**Data:** 229 stations in 171–175°E, 41–43.5°S from Beavan et al. (2016), ITRF2008 frame.

### Model selection

All three model selection criteria fail to identify a statistically preferred number of clusters. The gap statistic rises monotonically from k=1 to k=7, violating the unimodal structure that underpins the Tibshirani criterion. The reduced χ² decreases continuously from 126.6 (k=1) to 2.4 (k=8) without approaching unity, indicating that no finite rigid-block partition achieves an adequate kinematic fit. The sequential F-test yields p ≈ 0 for every transition k→k+1, meaning every additional cluster always improves fit significantly — there is no natural stopping point. Taken together, these diagnostics confirm that the MFS velocity field is characterised by *distributed* rather than block-like deformation: the five closely-spaced parallel faults produce a smooth kinematic gradient rather than discrete plate boundaries.

### A kinematic heterogeneity at the Kaikōura junction

Despite the absence of a statistically preferred block model, the clustering is not uninformative. At k=3, the algorithm consistently separates a cluster of 64 stations centred at 42.6°S, 172.9°E — directly overlying the junction of the Hope, Kaikōura, and Papatea fault strands that ruptured simultaneously in 2016. This cluster has a distinct Euler pole (6.8°N, 26.3°E, 0.53°/Ma) compared to the western (33.8°N, 21.5°E, 0.92°/Ma) and eastern (5.9°N, 39.1°E, 0.45°/Ma) clusters, reflecting a measurable kinematic gradient across the Kaikōura transition zone.

The contrast with Anatolia is instructive. In Anatolia the gap statistic peaks sharply at k=5, χ²_red approaches acceptable values, and the F-test stops — the block partition is statistically well-defined and physically unambiguous. In the MFS, the statistics say no partition is adequate, yet a kinematically anomalous sub-region still emerges at the future rupture site. This is not a contradiction: the MFS distributes strain continuously across its fault array, so no block model fits well globally, but local kinematic heterogeneity — the kind that enables multi-fault rupture — is nonetheless detectable. The method reveals a *gradient* rather than a *boundary*, and that gradient marks the hazard.

This distinction refines the scope of the method. Euler-vector clustering is most powerful as a block-boundary detector when deformation is genuinely block-like (Anatolia, SW Japan). In distributed systems it degrades gracefully to a kinematic anomaly detector — still useful for hazard, but requiring a different interpretive frame. Both behaviours are distinguishable from the model selection statistics alone, and both are scientifically meaningful.

**Comparison with geological slip rate estimates (to be completed):**
- Hope Fault: geological 23–25 mm/yr; geodetic (C1–C3 boundary at k=3): TBD
- Clarence Fault: geological 6–8 mm/yr; geodetic: TBD
- Awatere Fault: geological 6–9 mm/yr; geodetic: TBD

**Why this strengthens the Nature Comms case:** Anatolia and Marlborough are tectonically distinct — continental escape tectonics vs. subduction transition — yet both show the same pattern: multi-fault rupture zones correspond to kinematically anomalous regions recovered blind from pre-earthquake GPS clustering. In Anatolia the signal is a discrete block (statistically preferred k=5); in Marlborough it is a kinematic gradient (no preferred k, but anomalous cluster persists at the rupture site across k=3–5). Together they define the method's domain of applicability and demonstrate its generality.

### Relation to Takahashi & Hashimoto (2022)

Takahashi & Hashimoto (2022) applied both HAC and Euler pole clustering to the full New Zealand velocity field using the same Beavan et al. (2016) dataset. Their EPC analysis confirms the distributed-deformation character of the MFS: cluster boundaries do not coincide with mapped fault traces, and the MFS interior is characterised by high information entropy. This study takes that finding as a starting point and adds three dimensions absent from their work.

**Scale separation.** Takahashi & Hashimoto analyse all of New Zealand at once, so the MFS is always viewed through the AU–PA plate velocity contrast (~50 mm/yr). The Kaikōura kinematic anomaly is swamped at national scale by this dominant signal. Isolating the MFS subdomain (229 stations, 171–175°E) and applying the method locally recovers the within-domain gradient; the Kaikōura cluster emerges at k=3 with a distinct Euler pole and persists across k=3–5. Takahashi & Hashimoto do not perform a scale-separated MFS analysis and do not report the Kaikōura cluster.

**Formal model selection diagnostics.** Takahashi & Hashimoto use information entropy to assess cluster stability but do not compute gap statistics, χ²_red curves, or sequential F-test p-values for the MFS subdomain. The three-panel model selection figure here — gap monotonically rising, χ²_red never approaching unity, F-test p ≈ 0 for all k — formally characterises the MFS as a distributed-deformation system and provides a quantitative contrast with the Anatolian case where all three criteria converge on k=5. This transforms a qualitative observation ("distributed") into a falsifiable statistical statement.

**Retrospective hazard framing.** Takahashi & Hashimoto do not ask whether pre-2016 clustering localises the future rupture zone. Despite their EPC analysis using the same pre-earthquake data, the 2016 Kaikōura earthquake is not mentioned. The central question here — does Euler-vector clustering of pre-earthquake GPS detect the kinematic anomaly that enabled a surprise multi-fault rupture? — is entirely new. The affirmative answer for the Kaikōura junction, placed alongside the Kahramanmaraş result from Anatolia, constitutes the core scientific contribution.

---

## Nature Communications — Checklist before submission

- [ ] Marlborough case study complete
- [ ] Slip rate comparison table (geodetic vs geological) for all major boundaries
- [ ] Uncertainty quantification on Euler poles and derived slip rates (bootstrap or Monte Carlo)
- [ ] Formal statement of what "slow deforming" means in this context (strain rate threshold?)
- [ ] Seismic moment / recurrence interval from slip rates (one paragraph)
- [ ] Figure showing pre-earthquake block geometry → post-earthquake rupture overlay for both case studies
- [ ] Sentence addressing why prior hazard assessments missed these boundaries

---

## References

Beavan, J., et al. (2016). New Zealand GPS velocity field. *New Zealand Journal of Geology and Geophysics, 59*(1). doi:10.1080/00288306.2015.1112817

Haines, A., Dimitrova, L., Wallace, L. M., & Williams, C. A. (2015). *Enhanced surface imaging of crustal deformation: Obtaining tectonic force fields using GPS data*. Springer.

Litchfield, N. J., Van Dissen, R., Sutherland, R., Barnes, P. M., Cox, S. C., Norris, R., Beavan, J., Langridge, R., Villamor, P., Berryman, K., Stirling, M., Nicol, A., Nodder, S., Lamarche, G., Barrell, D. J. A., Pettinga, J. R., Little, T., Pondard, N., Mountjoy, J. J., & Clark, K. (2014). A model of active faulting in New Zealand. *New Zealand Journal of Geology and Geophysics, 57*(1), 32–56. doi:10.1080/00288306.2013.854256. Data set: https://doi.org/10.21420/W08T-TY11?x=y

Kılıç, B., & Özarpacı, S. (2022). Ensemble clustering in GPS velocities: A case study of Turkey. *Applied Sciences, 12*(24), 12636.

Özdemir, S., & Karslıoğlu, M. O. (2019). Soft clustering of GPS velocities from a homogeneous permanent network in Turkey. *Journal of Geodesy, 93*(8), 1171–1195.

Takahashi, A., & Hashimoto, M. (2022). Cluster analysis of dense GNSS velocity field reveals characteristics associated with regional tectonics in New Zealand. *Journal of Geophysical Research: Solid Earth, 127*, e2022JB024793. doi:10.1029/2022JB024793

Reilinger, R., McClusky, S., et al. (2006). GPS constraints on continental deformation in the Africa-Arabia-Eurasia continental collision zone. *Journal of Geophysical Research: Solid Earth, 111*(B5).

Savage, J. C. (2018). Euler-vector clustering of GPS velocities defines microplate geometry in southwest Japan. *Journal of Geophysical Research: Solid Earth, 123*(2), 1954–1968.

Simpson, R. W., Thatcher, W., & Savage, J. C. (2012). Using cluster analysis to organize and explore regional GPS velocities. *Geophysical Research Letters, 39*(18).
