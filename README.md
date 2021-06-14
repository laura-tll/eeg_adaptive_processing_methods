# eeg_adaptive_processing_methods

# Méthodes adaptatives pour le traitement de signaux EEG 

Ce dossier répertorie des codes Python et Matlab utilisés dans le cadre d'un stage de fin d'études ayant pour sujet l'amélioration des performances de classification d'une chaîne de traitement de signaux EEG. Les codes concernent trois méthodes : une chaîne de traitement pour les potentiels évoqués (ERP pour Event Related Potential), un programme d'implémentation d'une étape de filtrage spatial au moment de l'extraction des caractéristiques (Common Spatial Pattern), un programme d'implémentation d'un algorithme adaptatif pour la classification (AdaLDA pour Adaptive Linear Discriminant Analysis), et un programme d'analyse statistique pour tester la normalité d'une distribution (Test de Shapiro) et pour réaliser un T-test (stats_results_eeg.py).

### Description des programmes

- ERP : prétraitement pour données EEG découpées et classification avec 4 méthodes de validation (intra-sujet écologique, intra-sujet traditionnelle, inter-sujet écologique, inter-sujet traditionnelle).

- CSP : implémentation de l'algorithme Common Spatial Pattern pour bandes de fréquences (4 bandes ici thêta, alpha, bêta, gamma confondues mais possibilité de changer l'intervalles de fréquences pour l'adapter à la bande étudiée) avec 20 électrodes ou 32 électrodes + classification pour 2 méthodes de validation (inter-sujet écologique et inter-sujet traditionnelle), et programme de création de graphes pour comparaison des résultats. 

- AdaLDA : implémentation de l'algorithme Adaptive LDA (référence : AdaLDA https://github.com/linjunz/ADAM, code inspiré par l'article "High-dimensional Linear Discriminant Analysis : Optimality, Adaptive Algorithm, and Missing Data" de Tony Cai et Linjun Zhang) : classification pour 2 méthodes de validation (inter-sujet traditionnelle et intra-sujet écologique) et code pour calcul de l'accuracy de l'algorithme. 

### Data

Les données EEG utilisées sont celles récoltées par Gaganpreet Singh dans le cadre de sa thèse en cours ayant pour intitulé : "Interaction humain-machine : Intégration de l'état physiologique de l'opérateur humain dans la supervision".

