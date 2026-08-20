NASA’s Transiting Exoplanet Survey Satellite (TESS) produces millions of light curves each month. Within this massive and growing dataset lies potential signals for undiscovered exoplanets. Foundation models, large artificial intelligence systems that can be used for multiple scientific tasks, offer a way to parse through this dataset and discover new and rare planet candidates. However, sources of systematic noise can obscure subtle physical signals associated with exoplanets, making rare candidates difficult to detect. We are therefore developing a foundation model that disentangles the instrumental systematics from light curves to capture important physical properties. This basis can be used for anomaly detection to find rare types of signals such as disintegrating exoplanets or microlensing events. 

/<img width="1119" height="299" alt="Screenshot 2026-08-20 at 12 47 05 PM" src="https://github.com/user-attachments/assets/d89526d1-f0d8-4b48-8cd6-7afd3ecdcefa" />
This is the initial preprocessing pipeline for getting the systematics of the area around a light curve


<img width="1109" height="407" alt="Screenshot 2026-08-20 at 12 48 09 PM" src="https://github.com/user-attachments/assets/f8bc1cc6-2ad6-48c6-a036-1d0fba63b90e" />
This is the secondary pipeline that trains the physics side using classical JEPA, hoping to capture the true physics side

<img width="820" height="556" alt="Screenshot 2026-08-20 at 12 48 39 PM" src="https://github.com/user-attachments/assets/c9a753af-25e7-47c5-ac81-e01a8881ae6f" />

This is the entire pipeline at inference

Work is done in collaboration with MIT's Transiting Exoplanet Survey Satellite (TESS) Group
