# Whisper Algerian Dialect — Replicate Deployment

Déploiement du modèle **Whisper Algérien** fine-tuné sur **Replicate** pour DocCoworker.

## 📊 Specs du modèle

| Propriété | Valeur |
|---|---|
| **Base** | OpenAI Whisper-tiny |
| **Source HF** | `MohammedNasri/whisper-algerian-dialect` |
| **WER** | ~23% (Algérien pur) |
| **Langues** | Darija algérien + français code-switching |
| **Taille** | 0.2B params (léger) |
| **Framework** | HuggingFace Transformers + PyTorch |

## 🚀 Déployer sur Replicate

### Prérequis

- Compte GitHub
- Compte Replicate (https://replicate.com)
- Accès admin au repo GitHub

### Étapes

1. **Créer un nouveau repo GitHub**

   ```bash
   mkdir whisper-algerian-replicate
   cd whisper-algerian-replicate
   git init
   ```

2. **Copier les fichiers du déploiement**

   Copie `cog.yaml` et `predict.py` depuis ce dossier à la **racine** du repo GitHub :

   ```bash
   cp cog.yaml predict.py .
   git add cog.yaml predict.py
   git commit -m "feat: Deploy Whisper Algerian model to Replicate"
   git push origin main
   ```

3. **Connecter à Replicate**

   - Va sur https://replicate.com/create
   - Clique **"GitHub repository"**
   - Autorise Replicate à accéder à ton GitHub
   - Sélectionne le repo `whisper-algerian-replicate`
   - Replicate va **builder** l'image Docker automatiquement (~5-15 min)

4. **Obtenir la version API**

   Une fois construit avec succès, Replicate te donne :
   - **Model URL** : `replicate.com/username/whisper-algerian-replicate`
   - **Version ID** : UUID unique pour chaque version
   - **API endpoint** : `https://api.replicate.com/v1/predictions`

### Test initial

```bash
# Via cURL (une fois déployé)
curl -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "<VERSION_ID>",
    "input": {
      "audio": "https://example.com/audio.wav"
    }
  }'
```

---

## 🔗 Intégration dans DocCoworker

Une fois déployé sur Replicate, tu peux utiliser le modèle dans `src/lib/audio/replicate-transcribe.ts` :

```typescript
// Remplacer le version ID
const version = "<VOTRE_VERSION_ID_ALGERIAN>"; // Au lieu du Whisper large-v3

// Forcer la langue arabe (optionnel, le modèle la détecte déjà)
input: {
  audio: dataUri,
  language: "ar",
}
```

Ou créer une **nouvelle fonction** `transcribeWithAlgerianWhisper()` :

```typescript
export async function transcribeWithAlgerianWhisper(
  audioBuffer: Buffer,
): Promise<string> {
  const apiKey = process.env.REPLICATE_API_KEY;
  const version = process.env.REPLICATE_WHISPER_ALGERIAN_VERSION_ID!;
  
  const base64Audio = audioBuffer.toString("base64");
  const dataUri = `data:audio/webm;base64,${base64Audio}`;

  const createResp = await fetch("https://api.replicate.com/v1/predictions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      version,
      input: { audio: dataUri },
    }),
  });

  // ... (poll logic similar to current implementation)
}
```

Puis adapter `/api/consultations/[id]/transcribe` :

```typescript
if (provider === "algerian") {
  replicateText = await transcribeWithAlgerianWhisper(merged);
}
```

---

## ⚠️ Points importants

| Point | Détail |
|---|---|
| **Coût** | Déploiement gratuit sur Replicate ; tu paies à l'usage (~$0.001-0.01 par inference) |
| **Build time** | 5-15 min la première fois (CPU+GPU setup) |
| **GPU requis** | Oui (CUDA 12.0) — cf `cog.yaml` |
| **Proof-of-concept** | Le modèle n'a que 3 steps d'entraînement ; il peut s'améliorer avec plus de data |
| **Maintenance** | Le repo GitHub doit rester **public** pour que Replicate l'accède |

---

## 📝 Checklist avant déploiement

- [ ] Repo GitHub créé et pusté
- [ ] `cog.yaml` et `predict.py` à la racine
- [ ] Compte Replicate actif
- [ ] `REPLICATE_API_KEY` en env var (Replicate → Settings → API tokens)
- [ ] Build réussi sur Replicate (~5-15 min)
- [ ] Test initial avec cURL ou Replicate UI
- [ ] Version ID noté (`process.env.REPLICATE_WHISPER_ALGERIAN_VERSION_ID`)
- [ ] Intégration DocCoworker testée localement

---

## 🔍 Troubleshooting

| Erreur | Cause | Solution |
|---|---|---|
| "Model not found" | Repo privé ou mal connecté | Vérifier repo public + droits GitHub |
| "Build failed" | Dépendances Python manquantes | Ajouter à `python_packages` dans `cog.yaml` |
| "CUDA OOM" | Mémoire GPU insuffisante | Utiliser `float32` au lieu de `float16` dans `predict.py` |
| "Invalid model ID" | Typo dans `MohammedNasri/whisper-algerian-dialect` | Vérifier sur HuggingFace |

---

## 📚 Ressources

- **Modèle HF** : https://huggingface.co/MohammedNasri/whisper-algerian-dialect
- **Replicate Docs** : https://replicate.com/docs
- **Cog Framework** : https://github.com/replicate/cog
- **DocCoworker** : Voir `src/lib/audio/replicate-transcribe.ts`

---

*Créé pour DocCoworker MVP — Phase 2 optimisation Darija*
