# AppleSupport Intent Labeling Guide

This guide is for labeling `data/golden/golden_candidates.jsonl` into the
final `data/golden/golden_200.jsonl`. Read `data/golden/README.md` first for
overall sampling methodology; this file is the per-intent reference.

## How to label

For each candidate, read `conversation` (may be empty) + `customer_message`
(the message being classified) together, then set:

- **`intent`**: exactly one name from the list below.
- **`expected_action`**: `"auto_handle"` or `"escalate"` -- see "Escalation
  principle" below. Do not default to the intent's typical
  `escalation_guidance` without checking whether *this specific message*
  fits it.
- **`expected_reason`**: one concrete sentence specific to this message.
- **`label_notes`**: required whenever the example was ambiguous, multi-intent,
  or you overrode `suggested_intent` -- explain the call. Optional otherwise.

`suggested_intent` (and `suggested_intent_confidence`/`_margin`/`_second_choice`)
come from a frozen embedding-similarity classifier built only from
retrieval_pool data (see `src/evaluation/embedding_intent_classifier.py`).
**It is a starting point, not ground truth.** Phase D found real recall gaps
in an earlier, cruder keyword version of this idea -- verify every
suggestion against the actual text.

## Escalation principle

> The label should represent whether a reasonable support system should
> auto-handle the request **based on the information available in the
> conversation** -- not whether the issue is annoying, not whether the
> customer is upset, and not "escalate when in doubt."

Concretely: `auto_handle` is correct whenever a competent agent could give a
concrete, non-speculative, safe next step (a troubleshooting step, a factual
answer, a clarifying question) using only general product knowledge -- no
account access, no order lookup, no payment/refund decision, and no
information the historical AppleSupport conversations don't actually
support. `escalate` is correct when the request needs account/order access,
involves a financial commitment, is a known sensitive-safety topic, or the
available information is genuinely insufficient to do anything useful even
after asking a clarifying question would be redundant (e.g., the customer
already answered the obvious clarifying questions and the issue still isn't
resolvable without deeper access).

**Do not label every difficult or frustrating message as `escalate`.** A
short, vague, or frustrated message whose correct next step is simply "ask
what device/iOS version they have" is `auto_handle` -- asking a clarifying
question *is* the correct automated behavior, not a failure to automate.
Conversely, a calm, politely-worded request to check an order status is
still `escalate`, because the *information need*, not the tone, determines
the action.

## The 14 intents

### ios_update_general_complaint
**Definition:** Customer reports their device got worse/broke/behaves
strangely after an update, without naming a symptom that maps to a more
specific intent below.
**Include:** "phone's been terrible since the update," no named symptom;
first-contact messages that only imply *something* changed.
**Exclude:** any message that names a specific symptom (battery, freezing,
keyboard/autocorrect, connectivity, audio, media, photos, account) --
those go to the specific intent instead, even if the customer also blames
"the update" generally.
**Confusable with:** device_freezing_performance, battery_life_drain,
vague_frustration_needs_clarification, autocorrect_keyboard_bug.
**Examples:** "This is stupid. The new update for iPhone came out and I
can't even install an app" · "my phone keeps getting worse and worse with
these updates. WTH?!!!" · "Why does your new software have to make my
iphone6 obsolete."
**Escalation:** `auto_handle` with a clarifying question (device + iOS
version) by default. `escalate` only if the customer already gave those
details and it still doesn't match anything actionable, or this is a
repeat/unresolved complaint.

### autocorrect_keyboard_bug
**Definition:** The iOS 11.1-era bug where typing "I" produced a
box/question-mark glyph or "A".
**Include:** any mention of the letter-I/autocorrect/question-mark-box
symptom, regardless of exact phrasing (very high phrasing variety --
"I?", "eye", "boxes", "hieroglyphics" style complaints all count).
**Exclude:** general autocorrect complaints unrelated to the letter-I bug
(e.g. "autocorrect changed my word wrong" with no I/box/question-mark
symptom) -- label those `ios_update_general_complaint` instead.
**Confusable with:** ios_update_general_complaint.
**Examples:** "why does it auto correct to a box and question mark can you
tell me a fix" · "Can y'all seriously fix the issue of not being able to
type the letter I" · "FIX THIS I? SHIT NOOOOOW".
**Escalation:** `auto_handle` -- known bug, known fix (11.1.1+), no account
access needed. This is the taxonomy's clearest auto-handle case. `escalate`
only if the customer says the issue persists *after* updating to 11.1.1+.

### battery_life_drain
**Definition:** Battery draining unusually fast, poor battery life, or
charging problems.
**Include:** any battery-specific complaint, including implicit ones like
"phone died in 20 minutes" that don't use the literal word "battery."
**Exclude:** battery complaints that are secondary to a freezing/crashing
report where freezing is the primary named symptom -- use judgment on which
is the *primary* complaint (see multi-intent guidance below); physical
battery safety issues (swelling, heat, smoke) should still be
`battery_life_drain` as the intent, but `expected_action` must be `escalate`.
**Confusable with:** ios_update_general_complaint, device_freezing_performance.
**Examples:** "why is my battery draining so fast? Charged to 100%, then
about 20 minutes later was on 14%" · "my battery life on my phone has been
trash since the new update."
**Escalation:** `auto_handle` with standard battery-diagnostic guidance.
`escalate` for safety concerns (swelling/heat) or if standard steps were
already tried without success.

### device_freezing_performance
**Definition:** Device freezing, hanging, becoming unresponsive, or needing
a restart.
**Include:** freezing/hanging/crashing/unresponsive/slow-to-the-point-of-unusable.
**Exclude:** slowness attributed specifically to a named app/feature that
maps elsewhere (e.g. "Photos app locks up when cropping" -> could be
photos_storage_icloud_management if the storage/photos angle is primary;
use judgment on primary symptom).
**Confusable with:** ios_update_general_complaint, battery_life_drain.
**Examples:** "over 4k people reporting iOS 11 update is causing
freezing...and I am one of them." · "phone was completely frozen couldn't
restart phone."
**Escalation:** `auto_handle` with restart/force-quit/storage-check
guidance. `escalate` if a restart *and* a backup/restore step were already
tried without success, or data loss is reported.

### connectivity_network_messaging_issues
**Definition:** WiFi/Bluetooth toggling itself on, dropped cellular signal,
calls not connecting, iMessage/SMS problems, notifications not arriving.
**Include:** any network/radio/messaging-stack symptom.
**Exclude:** Bluetooth *audio quality/pairing* issues specifically (that's
audio_accessory_issues) -- if the complaint is about Bluetooth toggling
itself on/off, it's connectivity; if it's about audio cutting out over an
established Bluetooth connection, it's audio.
**Confusable with:** audio_accessory_issues, ios_update_general_complaint.
**Examples:** "Toggle from airplane mode turns on WiFi and Bluetooth" ·
"iPhone 6s Plus have lost the operator network. I cannot make any phone
call."
**Escalation:** `auto_handle` with standard network-reset guidance.
`escalate` if carrier-specific (may not be Apple's issue) or persists after
standard resets.

### audio_accessory_issues
**Definition:** EarPods/AirPods, speaker output, headphone jack/adapter, or
Bluetooth *audio* pairing/quality.
**Include:** no sound, one-sided audio, accessory hardware failure,
Bluetooth audio dropouts during use.
**Exclude:** Bluetooth connection toggling itself on/off with no audio
symptom -- that's connectivity_network_messaging_issues.
**Confusable with:** connectivity_network_messaging_issues.
**Examples:** "iPhone 6s's speaker has stopped working after updating" ·
"EarPods have broken three times this month."
**Escalation:** `auto_handle` with standard audio troubleshooting.
`escalate` for repeated hardware failure (a warranty/replacement case, which
itself should point toward order_purchase_retail_support handling).

### media_music_playback_issues
**Definition:** Apple Music/iTunes/Podcasts playback, library sync, missing
content, subscription access.
**Include:** anything about *consuming or managing* media content.
**Exclude:** App Store purchase/download mechanics for apps themselves
(app_store_purchase_issues) -- media vs. apps is the dividing line, not
"Apple service" broadly.
**Confusable with:** app_store_purchase_issues.
**Examples:** "Apple Music isn't working. Already a member but I can't
access the songs I've downloaded" · "why did the new update delete about
100 of my songs?"
**Escalation:** `auto_handle` with restart/re-sync guidance. `escalate` for
billing disputes (charged but content missing).

### app_store_purchase_issues
**Definition:** Downloading, updating, or purchasing apps through the App
Store; stalled downloads; in-app-purchase billing.
**Include:** app acquisition/installation/purchase mechanics.
**Exclude:** problems *using* an already-installed app that aren't about the
App Store itself (goes to whatever intent matches the app's function, or
ios_update_general_complaint if generic).
**Confusable with:** media_music_playback_issues.
**Examples:** "I purchased an in app upgrade and it's taken the money but
the upgrade hasn't been applied." · "updating xcode via appstore. 7-min
remaining 6-hr ago."
**Escalation:** `auto_handle` for stalled-download troubleshooting.
`escalate` for any billing dispute (charged without receiving the purchase).

### photos_storage_icloud_management
**Definition:** Photos app issues, iCloud Photo Library sync,
missing/duplicated pictures, device/iCloud storage management.
**Include:** anything about photo content or storage space.
**Exclude:** iCloud *login/account access* problems (account_apple_id_security)
-- storage/photos vs. account access is the dividing line, even though both
mention "iCloud."
**Confusable with:** account_apple_id_security.
**Examples:** "how do I get my pictures back they were deleted out of no
where?" · "storage almost full message is making me nuts!!!!"
**Escalation:** `auto_handle` with standard storage-management/sync
guidance. `escalate` if photos were unexpectedly deleted (possible account
compromise or sync bug needing real investigation).

### account_apple_id_security
**Definition:** Apple ID/iCloud login problems, password resets, suspicious
emails/calls claiming to be Apple, account access/closure requests.
**Include:** login failures, password reset requests, phishing verification
questions, "is this legit" questions, account lockouts.
**Exclude:** content/storage questions about an account the customer can
already access (photos_storage_icloud_management); order/purchase disputes
with no login component (order_purchase_retail_support).
**Confusable with:** photos_storage_icloud_management, order_purchase_retail_support.
**Examples:** "Can't login in [link] the login module doesn't work" · "hi
guys, just wondering if this email is genuine please?"
**Escalation:** `escalate` by default -- account access and phishing reports
are the sensitive-action category this project should never auto-handle,
regardless of classifier confidence. The one `auto_handle` exception: simply
pointing to a public "how to identify phishing" article when the customer
is asking whether something looks legit, not requesting an account change.

### mac_macos_issues
**Definition:** Mac hardware or macOS software problems (not iOS/iPhone).
**Include:** any Mac-specific hardware or macOS-specific software issue.
**Exclude:** iOS/iPhone issues even if the customer also owns a Mac and
mentions it in passing.
**Confusable with:** ios_update_general_complaint.
**Examples:** "after updating to High Sierra, finder windows glitch and
crash continuously." · "Mac OS10.13.1 SUCKS!!!! Dead machine after sleep
twice!!!"
**Escalation:** `auto_handle` with standard macOS troubleshooting (restart,
check updates, SMC reset). `escalate` for suspected known hardware defects
(route toward warranty handling).

### apple_watch_issues
**Definition:** Apple Watch hardware, watchOS, or pairing/connectivity
problems specific to the Watch.
**Include:** anything Watch-specific.
**Exclude:** general fitness/health app questions unrelated to the Watch
device itself.
**Confusable with:** none identified in Phase D.
**Examples:** "had my Apple Watch just over a week and the band is
delaminating already?" · "just bought a apple watch series 3 ... cant
connect watch to messenger."
**Escalation:** `auto_handle` with standard Watch troubleshooting/pairing
guidance. `escalate` for hardware defects (band delamination, burns) toward
warranty handling.

### order_purchase_retail_support
**Definition:** Order status/delivery, in-store purchase issues,
warranty/AppleCare coverage questions, physical defects noticed at
unboxing, general retail/support-experience dissatisfaction.
**Include:** anything requiring order/purchase/warranty lookup or
determination.
**Exclude:** account login problems with no order component
(account_apple_id_security).
**Confusable with:** account_apple_id_security.
**Examples:** "My iPhone X order is \"In Progress\". What does that mean?" ·
"unboxed new iPhone X and noticed this scratch on edge of the metal."
**Escalation:** `escalate` by default -- order status, replacements, and
warranty determinations need account/order access and involve a real
financial/logistics commitment this project should never auto-promise.

### vague_frustration_needs_clarification
**Definition:** Generic dissatisfaction or venting directed at the brand,
with no identifiable product, symptom, or actionable request.
**Include:** pure venting ("what's going on", swearing with no specifics),
off-topic content that got pulled into a support thread, messages too short
or vague to classify further.
**Exclude:** anything with even one identifiable symptom or product keyword
-- prefer the specific intent over this one whenever there's a real signal
to use.
**Confusable with:** ios_update_general_complaint (the other residual
bucket).
**Examples:** "What da fuck [brand]" · "I just wanna know what's going on
here."
**Escalation:** `auto_handle` -- the correct automated behavior is a
clarifying question, which counts as auto-handling, not a failure to
automate. `escalate` only if surrounding conversation context (not just
this message) already makes escalation obvious (e.g. this vague message
follows a clear security/order complaint upstream).

## Multi-intent messages

When a message genuinely raises more than one issue (e.g. "my battery is
draining fast AND I can't login to iCloud"), choose the intent that
determines the **correct next action** -- not necessarily whichever symptom
is mentioned first. If two intents would lead to genuinely different actions
(e.g. one auto-handle, one escalate), prefer the one that changes the
action outcome and document the other in `label_notes`. Record the
secondary issue explicitly, e.g. `label_notes: "primary=battery_life_drain;
secondary mention of icloud login, not the main ask"`.

## Ambiguous messages

If genuinely torn between two intents, pick the best-supported one (more
keyword/context matches, or the one a real AppleSupport agent's likely reply
would address first) and write `label_notes` explaining the call, e.g.
`label_notes: "ambiguous between mac_macos_issues and
device_freezing_performance; labeled freezing since that's the actionable
symptom, Mac is just the device"`.
