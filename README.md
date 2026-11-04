# Lightweight Continual Learning for DDoS Attack Detection in IoT Networks
Masterarbeit: Design and Optimization of a Lightweight Continual Learning Model for DDoS Attack Detection in the Internet of Things


## Forschungsprojekt – Masterarbeit

Dieses Repository enthält die Implementierung des im Rahmen einer Masterarbeit entwickelten **leichtgewichtigen Continual-Learning-Frameworks zur Erkennung von DDoS-Angriffen in IoT-Netzwerken**.

Das Ziel des Projekts besteht darin, ein Machine-Learning-Modell zu entwickeln, das nicht ausschließlich statisch auf einem einzelnen Trainingsdatensatz trainiert wird, sondern sich schrittweise an neue Netzwerkdaten und veränderte Angriffsmuster anpassen kann.

Ein besonderer Schwerpunkt liegt dabei auf der Kombination von **Elastic Weight Consolidation (EWC)**, einem **importance-basierten Replay Buffer** und einer **Conflict-Aware Replay-Strategie**. Dadurch soll ein ausgewogenes Verhältnis zwischen dem Erlernen neuer Angriffsmuster und der Bewahrung bereits erworbenen Wissens erreicht werden.

---

## 📌 Forschungsziel

IoT-Netzwerke erzeugen kontinuierlich große Mengen heterogener Netzwerkdaten. Gleichzeitig verändern sich Angriffsmuster im Laufe der Zeit. Ein klassisches Machine-Learning-Modell, das einmalig auf einem festen Datensatz trainiert wurde, kann daher bei neu auftretenden Angriffen an Leistungsfähigkeit verlieren.

Das vorliegende Projekt untersucht einen **Continual-Learning-Ansatz**, bei dem das Modell Daten schrittweise als aufeinanderfolgende Tasks verarbeitet.

Dabei werden insbesondere drei zentrale Herausforderungen adressiert:

* **Catastrophic Forgetting** beim Lernen neuer Tasks
* Auswahl relevanter historischer Daten unter begrenztem Speicher
* mögliche Konflikte zwischen alten und neuen Lernzielen

Die zentrale Idee besteht darin, diese Komponenten nicht isoliert zu betrachten, sondern ihre Wechselwirkungen innerhalb eines gemeinsamen Lernprozesses zu berücksichtigen.

---

# 🧠 Methodischer Ansatz

Das Framework basiert auf einem mehrkomponentigen Continual-Learning-System:

```text
                    ┌─────────────────────────┐
                    │     Netzwerkdaten       │
                    │   IoT / DDoS Traffic    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Continual DataLoader  │
                    │  Task-by-Task Streaming  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Base MLP          │
                    │     Neural Network      │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 │               │                │
                 ▼               ▼                ▼
          ┌────────────┐  ┌──────────────┐  ┌───────────────┐
          │    EWC     │  │ Importance   │  │ Adaptive      │
          │            │  │ Replay       │  │ Trigger       │
          └─────┬──────┘  └──────┬───────┘  └───────┬───────┘
                │                │                  │
                │                ▼                  │
                │        ┌───────────────┐          │
                │        │ Conflict-Aware│          │
                │        │    Replay     │          │
                │        └───────┬───────┘          │
                │                │                  │
                └────────────────┼──────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Continual Learning    │
                    │       Controller         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Updated Detection Model │
                    └─────────────────────────┘
```

---

# 🔄 Continual Learning

Im Gegensatz zu einem klassischen Offline-Trainingsprozess werden die Daten in diesem Framework als aufeinanderfolgende **Tasks bzw. Daten-Chunks** verarbeitet.

Nach jedem Task wird das Modell aktualisiert, ohne dass sämtliche historischen Trainingsdaten erneut vollständig verwendet werden müssen.

Dadurch entsteht ein Lernprozess der Form:

```text
Task 1 → Task 2 → Task 3 → ... → Task n
  │        │        │              │
  ▼        ▼        ▼              ▼
Learn    Adapt    Adapt          Adapt
```

Das Modell soll dabei sowohl neue Informationen aufnehmen als auch bereits erlerntes Wissen erhalten.

---

# 🛡️ Elastic Weight Consolidation (EWC)

Zur Reduzierung des **Catastrophic Forgetting** wird Elastic Weight Consolidation eingesetzt.

EWC identifiziert Parameter, die für bereits gelernte Tasks besonders wichtig sind. Diese Parameter werden bei späteren Updates stärker vor großen Veränderungen geschützt.

Die Loss-Funktion wird entsprechend um einen Regularisierungsterm erweitert:

```text
L_total = L_task + (λ / 2) Σ_i F_i (θ_i − θ_i*)²
```

Dabei bezeichnet:

* `L_task` – Loss des aktuellen Tasks
* `λ` – Stärke der EWC-Regularisierung
* `F_i` – Fisher Information des Parameters `i`
* `θ_i` – aktueller Parameterwert
* `θ_i*` – gespeicherter Parameterwert des vorherigen Lernzustands

Im implementierten System wird die Fisher Information nach den jeweiligen Lernschritten aktualisiert und für die nachfolgenden Tasks verwendet.

---

# 🧠 Importance-Based Replay

EWC schützt wichtige Modellparameter. Replay adressiert dagegen die Datenperspektive des Continual Learning.

Da nicht sämtliche historischen Daten dauerhaft gespeichert werden können, verwendet das Framework einen begrenzten Replay Buffer.

Die gespeicherten Samples werden nicht ausschließlich zufällig ausgewählt. Stattdessen wird jedem Sample ein **Importance Score** zugewiesen.

Ein Sample erhält beispielsweise eine höhere Bedeutung, wenn:

* der aktuelle Loss hoch ist,
* die Modellunsicherheit hoch ist,
* das Modell das Sample falsch klassifiziert.

Der verwendete Importance Score basiert auf:

```text
Importance =
    0.4 × normalized loss
  + 0.4 × normalized entropy
  + 0.2 × misclassification indicator
```

Die Samples werden anschließend abhängig von ihrer effektiven Bedeutung priorisiert.

---

# ⚔️ Conflict-Aware Replay

Ein zentraler Bestandteil des Frameworks ist die Erweiterung des Importance-based Replay um eine **Conflict-Aware-Komponente**.

Nicht jedes wichtige historische Sample unterstützt automatisch das Lernen des aktuellen Tasks. Ein historisches Sample kann gleichzeitig einen starken Lernkonflikt mit dem aktuellen Optimierungsziel verursachen.

Der Conflict Score basiert auf dem Verhältnis zwischen Gradienten des aktuellen Lernprozesses und den durch EWC geschützten Parameterbereichen.

Vereinfacht wird dabei untersucht, ob eine Aktualisierung des Modells gleichzeitig:

1. die Anforderungen des aktuellen Tasks verbessert und
2. bereits gelerntes Wissen gefährdet.

Die effektive Priorität eines Samples wird anschließend unter Berücksichtigung des Konflikts angepasst.

```text
Effective Weight
        =
Importance Score
        ×
Conflict Penalty
```

Die Conflict Penalty wird temperaturgesteuert berechnet:

```text
Conflict Penalty = exp(−Conflict Score / T)
```

Dadurch kann das Replay-System zwischen **hoher Informationsrelevanz** und **potenziell schädlichen Lernkonflikten** unterscheiden.

---

# 🌡️ Adaptive Trigger

Das Framework verarbeitet nicht zwangsläufig jeden zusätzlichen Mechanismus bei jedem Lernschritt mit gleicher Intensität.

Dafür besitzt das System eine adaptive Trigger-Komponente.

Unter anderem kann die Veränderung der Modellunsicherheit über die Zeit überwacht werden.

Für den Entropy Trigger wird die Veränderung zwischen einer Baseline und der aktuellen Entropie betrachtet:

```text
(H_recent − H_baseline) / H_baseline > δ
```

Wird der definierte Schwellenwert überschritten, kann dies auf eine Veränderung der Datenverteilung oder eine zunehmende Unsicherheit des Modells hinweisen.

Je nach Konfiguration können unterschiedliche Trigger-Strategien verwendet werden:

* Entropy-based Trigger
* Drift-based Trigger
* Performance-based Trigger
* Hybrid Trigger

---

# 🏗️ Systemarchitektur

Die Implementierung ist modular aufgebaut, sodass die einzelnen Continual-Learning-Komponenten unabhängig voneinander untersucht und konfiguriert werden können.

```text
project/
│
├── config/
│   └── default_config.yaml
│
├── data/
│   └── data_loader.py
│
├── models/
│   └── base_model.py
│
├── methods/
│   ├── ewc.py
│   ├── replay.py
│   └── conflict_aware_replay.py
│
├── trigger.py
├── controller.py
│
├── core/
│   └── system_builder.py
│
├── utils/
│   └── config_loader.py
│
└── main.py
```

### Komponenten

| Komponente                 | Funktion                                     |
| -------------------------- | -------------------------------------------- |
| `data_loader.py`           | Laden, Skalieren und Streamen der Daten      |
| `base_model.py`            | Definition des neuronalen Basis-Modells      |
| `ewc.py`                   | Elastic Weight Consolidation                 |
| `replay.py`                | Importance-based Replay Buffer               |
| `conflict_aware_replay.py` | Konfliktbewusstes Replay                     |
| `trigger.py`               | Adaptive Trigger-Mechanismen                 |
| `controller.py`            | Steuerung des Continual-Learning-Prozesses   |
| `system_builder.py`        | Aufbau und Initialisierung des Gesamtsystems |
| `config_loader.py`         | Laden und Verarbeiten der Konfiguration      |
| `main.py`                  | Einstiegspunkt für die Ausführung            |

---

# 🤖 Base Model

Als Basismodell wird ein leichtgewichtiges **Multilayer Perceptron (MLP)** verwendet.

Die Architektur besteht aus mehreren vollständig verbundenen Schichten mit:

* Linear Layers
* Batch Normalization
* ReLU Activation
* Dropout
* Output Layer

Die aktuelle Konfiguration verwendet:

```text
Input
  ↓
Linear(128)
  ↓
BatchNorm
  ↓
ReLU
  ↓
Dropout
  ↓
Linear(64)
  ↓
BatchNorm
  ↓
ReLU
  ↓
Dropout
  ↓
Linear(32)
  ↓
BatchNorm
  ↓
ReLU
  ↓
Dropout
  ↓
Output
```

Der modulare Aufbau ermöglicht es, die Architektur für weitere Experimente anzupassen.

---

# 📊 Verwendete Datensätze

Das Framework ist für die Verarbeitung von Netzwerkverkehrsdaten und insbesondere für DDoS-Erkennung in IoT-Umgebungen ausgelegt.

Im Rahmen der Experimente werden unter anderem folgende Datensätze berücksichtigt:

* **CIC-IoT2023**
* **CIC-DDoS2019**

Die Daten werden vor dem Training verarbeitet und skaliert. Die Skalierungsparameter werden entsprechend dem Continual-Learning-Szenario kontrolliert verwaltet, um Datenleakage zwischen den Tasks zu vermeiden.

---

# ⚙️ Konfiguration

Die wichtigsten Parameter des Experiments werden zentral in einer YAML-Konfigurationsdatei verwaltet.

Beispiel:

```yaml
training:
  batch_size: 64
  chunk_size: 500
  epochs_per_task: 10
  learning_rate: 0.001

model:
  hidden_layers:
    - 128
    - 64
    - 32
  dropout: 0.3

ewc:
  enabled: true
  lambda: 100.0
  fisher_sample_size: 200

replay:
  enabled: true
  buffer_size: 500
  batch_ratio: 0.3
  strategy: importance
  temperature: 1.0

conflict:
  enabled: true
  threshold: 0.5
  temperature: 1.0
  force_high_conflict_ratio: 0.1

trigger:
  type: entropy
  window_size: 100
  threshold: 0.15
```

Dadurch können unterschiedliche experimentelle Konfigurationen durchgeführt werden, ohne die Implementierung der einzelnen Methoden verändern zu müssen.

---

# 🚀 Installation

## Voraussetzungen

Empfohlen werden:

* Python 3.10+
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* PyYAML

Installation der Abhängigkeiten:

```bash
git clone https://github.com/ZenshiroDarkuOwO/conflict_aware_cl.git
cd conflict_aware_cl
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# oder: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

# ▶️ Ausführung

Nach der Konfiguration des Datensatzpfads und der gewünschten Experimentparameter kann das Framework über den zentralen Einstiegspunkt gestartet werden:

```bash
python main.py
```

Die einzelnen Komponenten werden entsprechend der YAML-Konfiguration initialisiert und anschließend durch den Continual-Learning-Controller orchestriert.

---

# 📈 Evaluation

Die Evaluation berücksichtigt nicht ausschließlich die klassische Klassifikationsleistung.

Neben der Modellgüte werden insbesondere Continual-Learning-relevante Kriterien betrachtet:

| Metrik       | Bedeutung                        |
| ------------ | -------------------------------- |
| Accuracy     | Klassifikationsleistung          |
| Forgetting   | Verlust zuvor erlernten Wissens  |
| Memory Usage | Speicherbedarf                   |
| Latency      | Inferenz- bzw. Verarbeitungszeit |

Damit wird eine ganzheitlichere Bewertung des Systems ermöglicht, insbesondere unter den Anforderungen ressourcenbeschränkter IoT- und Edge-Umgebungen.

---

# 🔬 Forschungsbeitrag

Der Schwerpunkt der Arbeit liegt auf der Untersuchung der **Interaktion zwischen verschiedenen Continual-Learning-Mechanismen**.

Während EWC hauptsächlich den Schutz wichtiger Modellparameter übernimmt, konzentriert sich Replay auf die Auswahl relevanter historischer Daten.

Die Conflict-Aware-Erweiterung versucht, beide Perspektiven miteinander zu verbinden:

```text
                Continual Learning
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
       EWC                      Replay
   Parameterseite             Datenseite
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
             Conflict Awareness
                       │
                       ▼
              Adaptive Learning
```

Das Ziel ist nicht lediglich die Kombination bestehender Verfahren, sondern die Untersuchung, wie deren Wechselwirkungen genutzt werden können, um ein robustes und zugleich ressourceneffizientes Continual-Learning-System für die DDoS-Erkennung zu entwickeln.

---

# 🎓 Kontext der Masterarbeit

Dieses Repository stellt den Implementierungsteil einer Masterarbeit mit dem Titel

> **„Design and Optimization of a Lightweight Continual Learning Model for DDoS Attack Detection in the Internet of Things“**

dar.

Die Implementierung dient der experimentellen Untersuchung der entwickelten Methodik und ermöglicht eine reproduzierbare Durchführung der beschriebenen Experimente.

---

# 📌 Projektstatus

**Status:** Abgeschlossen / Forschungsprototyp

Die Implementierung befindet sich im Rahmen der wissenschaftlichen Evaluation. Konfigurationsparameter, Modellarchitektur und einzelne Komponenten können im Verlauf weiterer Experimente angepasst und erweitert werden.

---

# 📚 Wissenschaftliche Grundlagen

Das Projekt basiert unter anderem auf Forschungsarbeiten zu:

* Continual Learning
* Catastrophic Forgetting
* Elastic Weight Consolidation
* Experience Replay
* DDoS Detection
* IoT Security
* Lightweight Machine Learning
* Edge Computing

Eine vollständige Darstellung der theoretischen Grundlagen und der verwendeten Literatur befindet sich in der zugehörigen Masterarbeit.

---

# 📄 Lizenz

Dieses Projekt steht unter der MIT Lizenz.

Die verwendeten Datensätze (z. B. CIC-IoT2023) unterliegen ihren eigenen Nutzungsbedingungen.

---

## 👤 Autor 
Zenshiro
**Masterarbeit – Forschung und Implementierung**

Dieses Repository dokumentiert die technische Umsetzung und experimentelle Grundlage der zugehörigen Masterarbeit.
