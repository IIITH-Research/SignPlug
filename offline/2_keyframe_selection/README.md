# 2. Sparse keyframe selection (ISL keyframe annotator)

A desktop app for marking keyframes in ISL dictionary videos. It hands you the videos in the
order of the prioritized annotation sheet, downloads each one from the lab NAS when you need it,
and uploads every finished annotation to the shared annotation machine, so everyone works from
the same list.

For each video you mark:

- **Start** and **End**: where the sign itself begins and ends (the interpolation range).
- **Keyframes**: the frames inside that range that define the sign's motion.

## What you need

- Windows, with **Python 3** from [python.org](https://www.python.org/downloads/). Tick
  *"Add python.exe to PATH"* during install.
- The **IIIT network** (on campus or VPN). The servers can't be reached from outside.
- Access to the two servers:
  - the **shared annotation machine** password, from Sharif (needed once, during setup)
  - the **NAS** (`sl_cvit`) password, from Sharif (videos are downloaded from there; the app
    asks for it once per session). Without it, setup prints a request with your computer's SSH
    key for the NAS admin. Key login needs the admin to give the `sl_cvit` account a home
    directory first, since it currently has none.
- Two private files, also from Sharif. They are deliberately not in git because this repository
  is public:
  - `servers.json`: the server addresses and folders
  - `annotation_queue.csv`: the prioritized sheet

## Getting started (once)

1. Clone the repository. Use a short folder path: Windows refuses paths longer than 260 characters.

   ```
   cd C:\
   git clone https://github.com/sharif-commits/ISL-AVATAR-PROJECT.git
   ```

2. Copy `servers.json` and `annotation_queue.csv` into the app folder:

   ```
   ISL-AVATAR-PROJECT\offline\2_keyframe_selection\
   ```

3. Double-click **`start.bat`** in that folder. The first run:
   - creates a Python environment and installs OpenCV and Pillow (a minute or two)
   - asks for the **shared machine password once** and sets up key login for this computer, so
     uploads never ask again
   - downloads the list of already-annotated videos
   - checks the NAS. Without a key there, it prints a request for the NAS admin; you can ignore
     it if you have the NAS password. (Print it again any time with
     `.\.venv\Scripts\python server_videos.py key`.)

   Then the app opens. From now on, `start.bat` just opens the app. The first download each
   session asks for the NAS password.

## Daily routine

We work on alternate days, so only one person annotates at a time.

1. Run `start.bat`. On start-up the app syncs with the shared machine (see the **Shared:** line),
   so everything the other annotator finished is skipped.
2. Click **⏭ Next in Queue**. The first download each session asks for the NAS password. The
   video opens, and the two after it download in the background.
3. Play or step through the video and mark:
   - **🟢 Mark Start** at the first frame of the sign
   - **🔴 Mark End** at the last frame of the sign
   - **🟡 Mark Keyframe** on each key pose in between
4. Click **💾 Save Metadata**. It saves on this computer and uploads the video and its
   metadata to the shared machine. The **Upload:** line shows ✓ when it's done.
5. Click **⏭ Next in Queue** and repeat.

To fix a video you already saved, select it in the dropdown, change the marks and save again.
The new save replaces the old one on the shared machine.

## The screen

**Top row**

| Control | What it does |
|---|---|
| Video dropdown | Sheet videos only. Videos still to annotate come first, in priority order; finished ones (●) go to the end. `⬇` means not downloaded yet, and selecting it downloads it. |
| ⏭ Next in Queue | Opens the next video to annotate after the current one. |
| ☁ Fetch from Server | Opens the full sheet: search by word or file name, tick *Pending only*, select one or more videos to download. |

**Active File Info (right)**

| Line | Meaning |
|---|---|
| Video | The file that's open. |
| Sheet | Its rank, word and priority, and whether it's annotated. `≈ same sign as X` means it's skipped because that sign is already done (see below). |
| Upload | Result of the last upload: ✓ done, ✗ failed (the save is still on your computer). |
| Shared | Result of the start-up sync with the shared machine. |

**Marks in the Fetch from Server list:** ● annotated · ≈ same sign already annotated ·
✓ downloaded · `·` on the NAS · ✗ not in the archive

## Which videos are skipped

A sheet video doesn't come up in the queue when it's already annotated:

- someone saved it with this app, or
- the same video is in the shared annotated set. Names are compared ignoring case and
  punctuation, so `Area_Sign_2` = `Area_(Sign_2).mp4` and `baby` = `Baby.mp4`.

It's also skipped when the **same sign** is already annotated under another name:

- **Numbered variants**: `Support_(Sign_2)` counts as done when `Support` is done.
- **Synonym entries** listed in `same_sign_aliases.csv`, for example `Own_Mine` = `Own` and
  `Want_Desire_Wish_For` = `want`. The list was checked by hand. Compounds like `Heart_Attack`
  and ambiguous words like `Lie_False` or `Cry_Sob` were left in the queue on purpose. To skip
  another entry, add a line `sheet_video,same_sign_as`.

Variants with a qualifier are different signs and stay in the queue, for example
`Cold_(for_food)` and `Banner_(Gujarat)`.

## Where things are saved

| What | Where |
|---|---|
| Downloaded videos | `ISL-AVATAR-PROJECT\dataset\` |
| Your saved annotations | `meta-data\<video>_meta-data.json` in the app folder |
| Uploaded copies | the shared machine's video and metadata folders (set in `servers.json`) |

A saved annotation looks like this:

```json
{
    "video_name": "Assistance.mp4",
    "interpolate_start": 12,
    "interpolate_end": 64,
    "keyframes": [12, 20, 31, 47, 64],
    "base_word": "Assistance",
    "web_rank": 8
}
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `start.bat` says servers.json / annotation_queue.csv are missing | Copy both files into the app folder (Getting started, step 2). |
| "Python 3 was not found" | Install Python from python.org with *Add python.exe to PATH* ticked, then run `start.bat` again. |
| App asks for the NAS password and you don't have it | Ask Sharif for the `sl_cvit` password. Or click **Cancel**: the app copies a key request to the clipboard for the NAS admin, who must also give `sl_cvit` a home directory. |
| `authorize-nas` says the account has no home directory | The password was right, but `sl_cvit` can't hold SSH keys yet. Keep using the password, or ask the admin to give the account a home directory. |
| Password box keeps coming back | The NAS password was wrong; type it again. |
| **Upload: ✗** | Your save is safe on your computer. When the network is back, open a terminal in the app folder and run `.\.venv\Scripts\python server_videos.py push`. Add `--dry-run` first to see what it will send. |
| **Shared: ✗ offline** | The shared machine couldn't be reached (not on the IIIT network?). The app uses the last sync; restart once you're connected. |
| A video says ✗ (not in archive) | That sheet row has no video on the NAS. Skip it. |
| Setup failed half-way | Fix the error it printed, delete `.setup_done` if it exists, and run `start.bat` again. |

Getting updates: `git pull` in the repo folder. Your annotations aren't affected; they're on
the shared machine and in `meta-data\`.

## Command line (optional)

Run these from the app folder with `.\.venv\Scripts\python server_videos.py <command>` (works in
PowerShell and cmd; if you use your own Python environment, just `python server_videos.py <command>`):

| Command | What it does |
|---|---|
| `progress` | How much is done: sheet coverage, your recent saves and whether they uploaded, anything skipped over, what's next |
| `queue` | Progress, plus the next videos to annotate |
| `fetch --next 20` | Download the next 20 videos ahead of time |
| `fetch --ranks 1-100` | Download sheet rows by rank |
| `search please` | Find a word in the sheet |
| `push [--dry-run] [--force]` | Upload saved annotations that aren't on the shared machine yet |
| `annotated --sync` | Re-sync the annotated set and list matches, variants and skipped synonyms |
| `index --refresh` | Re-list the NAS archive (cached in `remote_index.json`) |
| `check` | Test the NAS connection |
| `key` | Print the request for the NAS admin (your computer's SSH key) |
| `authorize-nas` | Authorize this computer's key on the NAS; whoever has the NAS password types it once (needs a home directory on the account) |
| `setup` | Run the first-time setup again |

## For developers

- `app.py`: the Tkinter app. `server_videos.py`: NAS downloads (videos are streamed out of the
  tar with `tar -xOf`, so nothing is extracted on the NAS), uploads, sync, queue order and
  same-sign matching.
- Server settings come from `servers.json` (template: `servers.example.json`). Environment
  variables override it; see the top of `server_videos.py`.
- The NAS tries key login first (the admin adds the annotator's key, see `key` /
  `authorize-nas`), then password login. The password is asked once per session, kept in
  memory, and passed to `ssh` through an `SSH_ASKPASS` helper, never written to disk.
- `annotated_raw.txt` and `annotated_remote_meta.txt` are caches rebuilt on every sync.

### Keyframe correction and interpolation pipeline

After keyframes are selected with `app.py` (stored in `meta-data/`):

1. Store the MediaPipe detections in `mediapipe_detections/`.
2. Load `app.js`. It shows only the keyframes that need correcting.
3. After correcting a video, store it in `keyframes_corrected/`.
4. In the notebook, run the cells in the order of the words of a sentence; a slerp function
   interpolates between words to build the sentence.
5. Store the interpolated result in `keyframes_corrected/`.
6. Run `app.js` again with the sentence file name to see the sentence animation.
