# Motive App Flow

## Navigation Flow (6 pages)

```
┌─────────────┐
│   LOGIN     │
│  (auth)     │
└──────┬──────┘
       │ sign in
       ▼
┌─────────────┐
│  PROJECTS   │◄──────────────────────────────────┐
│  (list)     │                                    │
└──────┬──────┘                                    │
       │ click project                             │ back
       ▼                                           │
┌─────────────┐                                    │
│  WORKSPACE  │────────────────────────────────────┘
│  (file list)│
└──────┬──────┘
       │
       ├── click file ──────────────────────────┐
       │                                         ▼
       │                          ┌──────────────────────────┐
       │                          │      FILE ANALYSIS       │
       │                          │                          │
       │                          │  NEW → SEGMENTED         │
       │                          │    → ANALYZING           │
       │                          │    → ANALYZED            │
       │                          │    → MATCHED             │
       │                          │                          │
       │                          └────────────┬─────────────┘
       │                                       │ back
       │◄──────────────────────────────────────┘
       │
       ├── "Set Rights" (per file or bulk select)
       │         │
       │         ▼
       │   ┌──────────────────────────────────┐
       │   │        RIGHTS SETUP              │
       │   │                                  │
       │   │  • Territory                     │
       │   │  • Medium                        │
       │   │  • Time period                   │
       │   │  • Exclusivity                   │
       │   │  • Live cost preview             │
       │   │  • "Apply to similar" (bulk)     │
       │   │                                  │
       │   └───────────────┬──────────────────┘
       │                   │ save
       │◄──────────────────┘
       │
       │  Once any file has rights + matches → draft licenses exist
       │
       └── "Review Delivery"
             │
             ▼
┌──────────────────────────────────────────────┐
│          PROJECT DELIVERY                     │
│                                               │
│  • All licenses grouped by file               │
│  • Per-file subtotals, project total          │
│  • Play matched tracks, video sync, A/B       │
│  • Adjust rights (links back to Rights Setup) │
│  • "Approve & Lock" → .dcs records            │
│  • "Download Package" / email to client       │
│  • Export (CSV, PDF)                          │
└───────────────────────────────────────────────┘
```

## File States

| State | Description | UI |
|-------|-------------|-----|
| **New** | Added but never analyzed | Waveform visible, prompt to segment |
| **Segmented** | Segments detected, not yet analyzed | Waveform with segments, prompt to analyze |
| **Analyzing** | Analysis in progress | Shimmer/progress indicator |
| **Analyzed** | Full results with cue cards | Tags may or may not be curated |
| **Matched** | Analyzed + track replacement found | Replacement track linked to cue(s) |

## Workspace File Card Badges (rights-related)

| State | Badge |
|-------|-------|
| No rights set | Grey "Set Rights" |
| Rights set, no matches | Rights summary, no cost |
| Rights + matches = draft licenses | Cost estimate shown |
| Locked | Lock icon, final cost |

## Licensing Model

**1 license = 1 matched track + 1 file + 1 rights scope**

Rights are per-file (not per-project) because different files may air on different media with different territorial scope. PROs require per-broadcast reporting.

**License lifecycle:** draft → approved → locked (hashed, immutable .dcs record)

**Project client contact:** name, email, company, role — can receive delivery package and sign/approve.
