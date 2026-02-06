# Flan-T5 (Textgenerierung)

## Projektbeschreibung

Dieses Repository dokumentiert den Einsatz des Sprachmodells **Flan-T5** im Rahmen der Bachelorarbeit im Studiengang Wirtschaftsinformatik an der Hochschule für Technik und Wirtschaft Berlin (HTW Berlin).

Ziel des Projekts ist die Erstellung einer **Nutzungs- und Leistungsdokumentation für GPU-gestützte Leihlaptops**. FLan-T5 dient dabei als praxisnahes Einstiegsszenario für Studierende sowie als Werkzeug zur **Leistungsbewertung der GPU-Hardware** unter realistischen Bedingungen.

---

## Art der Daten

1. **Eingabedaten**

   * Texteingaben und Prompts zur Textgenerierung
   * Konfigurationsparameter der Modellaufrufe
   * als *.PNG* Dateien

2. **Ausgabedaten**

   * Generierte Texte (Zusammenfassungen, Antworten, Transformationen)
   * als *.PNG* Dateien

3. **Logdaten**

   * Laufzeiten, Speicherverbrauch und GPU-Auslastung

**Sprache:** Deutsch / Englisch (modellbedingt)

**Methode:**
Experimentelle Leistungsbewertung durch Vergleich unterschiedlicher Promptkonfigurationen.

---

## Datenursprung

* **Autor:** Studierender der HTW Berlin, Wirtschaftsinformatik
* **Datenherkunft:** Testprompts wurden durch ChatGPT erstellt, Ergebnisse und Logs wurden selbst erhoben
* **Lizenz:**

  * **Code & Logs:** Apache License 2.0
  * **Textergebnisse:** Apache License 2.0

---

## Datenformate und -umfang

* Texte: *.PNG* (mehrere KB pro Prompt)
* Logs: *.csv* (1-2 KB)

---

## Qualitätssicherung

* Wiederholte Generierung mit identischen Prompts
* Vergleich von Antwortlängen und Laufzeiten
* Konsistente Dokumentation der Parameter

---

## Ordnerstruktur: **

```
/
├─ prompts/
├─ outputs/
├─ logs/
└─ README.md
```

---

## Hinweise zur Nachnutzung

Die Nutzung der Daten und Ergebnisse ist unter den Bedingungen der Apache License 2.0 zulässig.
