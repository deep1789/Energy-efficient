# Datasets and group structure

We use three public datasets that are frequently used to compare regression methods and that were obtained from designed experiments or repeated measurements. All three are available from the UCI Machine Learning Repository and mirrored on Kaggle; the exact files used are in the `data/` folder of the accompanying repository. Table 3 summarises their size, targets and group structure, and Fig. 2 visualises the grouping.

Table: Table 3. Datasets, targets and group structure. $G$ is the number of groups; $m$ is the number of rows per group; ICC is the intraclass correlation of the target (one-way ANOVA estimator, unbalanced design). SD is the standard deviation of the target.

| Dataset | Rows | Features | Target (unit) | Mean ± SD | $G$ | $m$ (min / median / max) | ICC of target |
|:---|---:|---:|:---|:---|---:|:---|---:|
| ENB2012 | 768 | 8 | Heating load (kWh/m²) | 22.31 ± 10.09 | 12 | 64 / 64 / 64 | 0.914 |
| ENB2012 | 768 | 8 | Cooling load (kWh/m²) | 24.59 ± 9.51 | 12 | 64 / 64 / 64 | 0.927 |
| Concrete | 1,030 | 8 | Compressive strength (MPa) | 35.82 ± 16.71 | 427 | 1 / 1 / 20 | 0.406 |
| Parkinson's | 5,875 | 19 | Total UPDRS (points) | 29.02 ± 10.70 | 42 | 101 / 141 / 168 | 0.935 |
| Parkinson's | 5,875 | 19 | Motor UPDRS (points) | 21.30 ± 8.13 | 42 | 101 / 141 / 168 | 0.919 |

![Fig. 2. Group structure of the three datasets. (a) ENB2012 heating load by building geometry; each point is one of 64 variants and the horizontal line marks the geometry mean. (b) Concrete: histogram of the number of rows per mix; 246 of 427 mixes occur once. (c) Parkinson's: distribution of total UPDRS per patient, sorted by patient mean. (d) Intraclass correlation of each target with respect to its group variable.](../figures/paper/fig02_structure.png){width=6.3in}

## ENB2012: simulated building geometries

The Energy Efficiency dataset contains 768 buildings simulated in Ecotect, all with the same volume (771.75 m³), materials and location, and differing in layout, glazing and orientation [@tsanas2012]. The eight inputs are relative compactness ($X_1$), surface area ($X_2$), wall area ($X_3$), roof area ($X_4$), overall height ($X_5$), orientation ($X_6$, four values), glazing area ($X_7$, four values: 0, 0.10, 0.25, 0.40 of floor area) and glazing-area distribution ($X_8$, six values). The two targets are the heating load $Y_1$ and cooling load $Y_2$. There are no missing values and no duplicate rows. Attributes $X_1$–$X_5$ take exactly twelve combinations, each shared by 64 rows (four orientations times sixteen glazing configurations); we call each combination a *geometry* (Table 4). Six geometries are 7 m high and six are 3.5 m high. The between-geometry share of the target variance is 0.91 (heating) and 0.93 (cooling). Because the data are produced by a deterministic simulation, there is no measurement noise: all variation within a geometry is explained by orientation and glazing.

Table: Table 4. The twelve ENB2012 geometries and their mean loads (kWh/m²) over the 64 variants.

| Geometry | $X_1$ | $X_2$ | $X_3$ | $X_4$ | $X_5$ | Heating | Cooling |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.98 | 514.5 | 294.0 | 110.25 | 7.0 | 27.65 | 29.22 |
| 2 | 0.90 | 563.5 | 318.5 | 122.50 | 7.0 | 31.63 | 33.82 |
| 3 | 0.86 | 588.0 | 294.0 | 147.00 | 7.0 | 28.55 | 30.91 |
| 4 | 0.82 | 612.5 | 318.5 | 147.00 | 7.0 | 25.56 | 28.03 |
| 5 | 0.79 | 637.0 | 343.0 | 147.00 | 7.0 | 38.61 | 40.24 |
| 6 | 0.76 | 661.5 | 416.5 | 122.50 | 7.0 | 35.66 | 36.41 |
| 7 | 0.74 | 686.0 | 245.0 | 220.50 | 3.5 | 11.89 | 14.81 |
| 8 | 0.71 | 710.5 | 269.5 | 220.50 | 3.5 | 12.04 | 15.04 |
| 9 | 0.69 | 735.0 | 294.0 | 220.50 | 3.5 | 12.39 | 15.24 |
| 10 | 0.66 | 759.5 | 318.5 | 220.50 | 3.5 | 12.82 | 15.87 |
| 11 | 0.64 | 784.0 | 343.0 | 220.50 | 3.5 | 16.62 | 20.23 |
| 12 | 0.62 | 808.5 | 367.5 | 220.50 | 3.5 | 14.28 | 15.24 |

The geometry grid has a one-dimensional backbone: relative compactness decreases and surface area increases monotonically from geometry 1 to 12, while height changes once, between geometries 6 and 7. A held-out geometry is therefore an interpolation if it lies between two training geometries of the same height and an extrapolation if it lies at an end of the grid (geometries 1, 6, 7 and 12) or if the height group is represented by few training geometries. We use this structure in Section 6.2.

## Concrete compressive strength

The dataset of Yeh contains 1,030 concrete samples with the quantities (kg/m³) of cement, blast-furnace slag, fly ash, water, superplasticizer, coarse aggregate and fine aggregate, the age of the specimen (1–365 days) and its compressive strength in MPa [@yeh1998]. There are no missing values; 25 rows are exact duplicates. Rows that share all seven ingredient quantities belong to the same mix and differ in age and strength; this grouping gives 427 mixes with a median size of 1 and a maximum of 20 rows. Of these, 246 mixes occur once, and 784 of the 1,030 rows belong to mixes that occur more than once (Fig. 2b). The between-mix share of strength variance is 0.41, much lower than in the other datasets, because strength depends strongly on age within a mix and on composition across mixes.

## Parkinson's telemonitoring

The Parkinson's Telemonitoring dataset contains 5,875 voice recordings from 42 people with early-stage Parkinson's disease in a six-month home-monitoring trial [@tsanas2010]. Each row contains the subject identifier, age, sex, the time since recruitment in days (`test_time`), the motor and total UPDRS scores [@fahn1987] interpolated to the recording date, and 16 voice measures (several variants of jitter and shimmer, noise-to-harmonics and harmonics-to-noise ratios, and the nonlinear measures RPDE, DFA and PPE). Patients contribute between 101 and 168 recordings. There are no missing values and no duplicate rows. Age and sex are constant within a patient and therefore identify the patient; `test_time` places a recording on the patient's disease trajectory. Between-patient differences account for 0.93 (total UPDRS) and 0.92 (motor UPDRS) of the variance (Fig. 2c, d). Unless stated otherwise we use all 19 features, which is the standard setup; Section 6.3 examines feature subsets.

## What the group structure implies

The three datasets span a range of dependence. ENB2012 has few, large and deterministic groups with an extremely high ICC; Parkinson's has few, large, noisy groups with an equally high ICC; Concrete has many small groups, nearly half of them singletons, and a moderate ICC. According to the theory in Section 3, we therefore expect severe inflation for flexible learners on the first two datasets and a mild effect on the third. The leaked fraction under random splitting is 100%, 75% and 100% for the three datasets (Section 6.1 and Eq. (12)).
