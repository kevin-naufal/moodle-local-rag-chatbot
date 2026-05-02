<?php
defined('MOODLE_INTERNAL') || die();

/**
 * Returns project path for RAG workspace.
 *
 * @return string
 */
function local_chatbot_get_project_path(): string {
    $path = trim((string)get_config('local_chatbot', 'projectpath'));
    if ($path === '') {
        $path = 'C:\\Users\\Kevin\\Downloads\\my-llm';
    }
    return $path;
}

/**
 * Returns python path for RAG runner.
 *
 * @return string
 */
function local_chatbot_get_python_path(): string {
    $path = trim((string)get_config('local_chatbot', 'pythonpath'));
    if ($path === '') {
        $path = 'C:\\Users\\Kevin\\Downloads\\my-llm\\.venv\\Scripts\\python.exe';
    }
    return $path;
}

/**
 * Returns runner filename.
 *
 * @return string
 */
function local_chatbot_get_runner_file(): string {
    $file = trim((string)get_config('local_chatbot', 'runnerfile'));
    if ($file === '') {
        $file = 'app/moodle_rag_runner.py';
    }
    return $file;
}

/**
 * Resolve runner script path with backward-compatible fallbacks.
 *
 * @return string
 */
function local_chatbot_resolve_runner_path(): string {
    $projectpath = local_chatbot_get_project_path();
    $configured = trim(local_chatbot_get_runner_file());
    $candidates = [];

    if ($configured !== '') {
        $normalized = ltrim(str_replace(['\\', '/'], DIRECTORY_SEPARATOR, $configured), DIRECTORY_SEPARATOR);
        $candidates[] = $projectpath . DIRECTORY_SEPARATOR . $normalized;
    }

    // Preferred location after project refactor.
    $candidates[] = $projectpath . DIRECTORY_SEPARATOR . 'app' . DIRECTORY_SEPARATOR . 'moodle_rag_runner.py';
    // Backward compatibility for older layout.
    $candidates[] = $projectpath . DIRECTORY_SEPARATOR . 'moodle_rag_runner.py';

    $seen = [];
    foreach ($candidates as $candidate) {
        if (isset($seen[$candidate])) {
            continue;
        }
        $seen[$candidate] = true;
        if (is_file($candidate)) {
            return $candidate;
        }
    }

    return $candidates[0];
}

/**
 * Returns data directory path.
 *
 * @return string
 */
function local_chatbot_get_data_path(): string {
    return local_chatbot_get_project_path() . DIRECTORY_SEPARATOR . 'data';
}

/**
 * Ensures data directory exists.
 *
 * @return void
 */
function local_chatbot_ensure_data_dir(): void {
    $datadir = local_chatbot_get_data_path();
    if (!is_dir($datadir)) {
        mkdir($datadir, 0777, true);
    }
}

/**
 * Lists uploaded PDF/TXT files from data directory.
 *
 * @return array
 */
function local_chatbot_list_uploaded_files(): array {
    local_chatbot_ensure_data_dir();
    $datadir = local_chatbot_get_data_path();
    $files = [];

    foreach (scandir($datadir) as $name) {
        if ($name === '.' || $name === '..') {
            continue;
        }

        $path = $datadir . DIRECTORY_SEPARATOR . $name;
        if (!is_file($path)) {
            continue;
        }

        $ext = strtolower(pathinfo($name, PATHINFO_EXTENSION));
        if ($ext !== 'pdf' && $ext !== 'txt') {
            continue;
        }

        $files[] = [
            'name' => $name,
            'size' => filesize($path),
            'modified' => filemtime($path),
        ];
    }

    usort($files, static function($a, $b) {
        return strcasecmp($a['name'], $b['name']);
    });

    return $files;
}

/**
 * Remove synced PDF/TXT files from chatbot data directory.
 *
 * @return void
 */
function local_chatbot_clear_data_dir_documents(): void {
    local_chatbot_ensure_data_dir();
    $datadir = local_chatbot_get_data_path();
    foreach (scandir($datadir) as $name) {
        if ($name === '.' || $name === '..') {
            continue;
        }
        $path = $datadir . DIRECTORY_SEPARATOR . $name;
        if (!is_file($path)) {
            continue;
        }
        $ext = strtolower(pathinfo($name, PATHINFO_EXTENSION));
        if ($ext === 'pdf' || $ext === 'txt') {
            @unlink($path);
        }
    }
}

/**
 * Build unique filename for data directory.
 *
 * @param string $basename
 * @param array $usednames
 * @return string
 */
function local_chatbot_unique_data_filename(string $basename, array &$usednames): string {
    $clean = clean_param($basename, PARAM_FILE);
    if ($clean === '') {
        $clean = 'material.pdf';
    }
    $name = pathinfo($clean, PATHINFO_FILENAME);
    $ext = pathinfo($clean, PATHINFO_EXTENSION);
    $candidate = $clean;
    $i = 2;
    while (isset($usednames[$candidate])) {
        $suffix = '-' . $i;
        $candidate = $ext !== ''
            ? ($name . $suffix . '.' . $ext)
            : ($name . $suffix);
        $i++;
    }
    $usednames[$candidate] = true;
    return $candidate;
}

/**
 * Quote command argument for Windows cmd.
 *
 * @param string $arg
 * @return string
 */
function local_chatbot_quote_arg(string $arg): string {
    $arg = str_replace('"', '\"', $arg);
    return '"' . $arg . '"';
}

/**
 * Detects whether model output is a generic fallback answer.
 *
 * @param string $answer
 * @return bool
 */
function local_chatbot_is_generic_fallback_answer(string $answer): bool {
    $normalized = core_text::strtolower(trim($answer));
    if ($normalized === '') {
        return true;
    }
    if ($normalized === 'sorry, i cannot provide an answer for that question yet.') {
        return true;
    }
    if (strpos($normalized, 'sorry, i cannot provide an answer for that question yet.') === 0) {
        return true;
    }
    return false;
}

/**
 * Runs Python RAG runner once and returns answer.
 *
 * @param string $question
 * @param string $mode
 * @return array
 */
function local_chatbot_run_rag_once(string $question, string $mode = 'auto'): array {
    $python = local_chatbot_get_python_path();
    $runner = local_chatbot_resolve_runner_path();
    $datadir = local_chatbot_get_data_path();
    $mode = core_text::strtolower(trim($mode));
    if (!in_array($mode, ['auto', 'general', 'general_raw'], true)) {
        $mode = 'auto';
    }

    if (!is_file($python)) {
        throw new Exception('Python executable not found: ' . $python);
    }
    if (!is_file($runner)) {
        throw new Exception(
            'Runner script not found. Checked path: ' . $runner .
            '. Please set runner file to app/moodle_rag_runner.py in local_chatbot settings.'
        );
    }

    $questionb64 = base64_encode($question);
    $cmd = local_chatbot_quote_arg($python) . ' ' .
        local_chatbot_quote_arg($runner) . ' --data-dir ' .
        local_chatbot_quote_arg($datadir) . ' --query-b64 ' .
        local_chatbot_quote_arg($questionb64) . ' --mode ' .
        local_chatbot_quote_arg($mode) . ' 2>&1';

    $output = [];
    $code = 0;
    exec($cmd, $output, $code);
    $raw = trim(implode("\n", $output));

    if ($code !== 0) {
        throw new Exception('RAG process failed: ' . $raw);
    }

    $jsonline = $raw;
    if (strpos($raw, "\n") !== false) {
        $lines = preg_split('/\r\n|\r|\n/', $raw);
        $jsonline = trim((string)end($lines));
    }

    $payload = json_decode($jsonline, true);
    if (!is_array($payload) || !array_key_exists('answer', $payload)) {
        throw new Exception('Invalid runner response: ' . $raw);
    }

    return [
        'answer' => (string)$payload['answer'],
        'sources' => isset($payload['sources']) && is_array($payload['sources']) ? $payload['sources'] : [],
    ];
}

/**
 * Runs Python RAG runner and retries once for long prompts if response is generic fallback.
 *
 * @param string $question
 * @return array
 */
function local_chatbot_run_rag(string $question): array {
    $result = local_chatbot_run_rag_once($question, 'auto');
    $normalizedquestion = core_text::strtolower(trim($question));
    $islongprompt = core_text::strlen(trim($question)) >= 80;
    $issimplegreeting = in_array($normalizedquestion, ['hi', 'hello', 'halo', 'hey'], true);

    if (!$issimplegreeting && $islongprompt && local_chatbot_is_generic_fallback_answer((string)$result['answer'])) {
        $result = local_chatbot_run_rag_once($question, 'auto');
    }

    return $result;
}

/**
 * Runs Python runner in general-LLM mode (without retrieval context).
 *
 * @param string $prompt
 * @param bool $rawanswer when true, suppress markdown normalization from runner
 * @return array
 */
function local_chatbot_run_llm_general(string $prompt, bool $rawanswer = false): array {
    $mode = $rawanswer ? 'general_raw' : 'general';
    return local_chatbot_run_rag_once($prompt, $mode);
}

/**
 * Detect whether a prompt is for structured assignment/practice draft generation.
 *
 * @param string $prompt
 * @return bool
 */
function local_chatbot_is_structured_generation_prompt(string $prompt): bool {
    $normalized = core_text::strtolower(trim($prompt));
    if ($normalized === '') {
        return false;
    }

    $markers = [
        'assignment title:',
        'learning objectives:',
        'instructions for students:',
        'question list:',
        'answer key:',
        'grading rubric:',
        'judul tugas:',
        'tujuan pembelajaran:',
        'instruksi untuk siswa:',
        'daftar soal:',
        'kunci jawaban:',
        'rubrik penilaian:',
    ];

    $hits = 0;
    foreach ($markers as $marker) {
        if (strpos($normalized, $marker) !== false) {
            $hits++;
        }
    }
    if ($hits >= 3) {
        return true;
    }

    if (strpos($normalized, 'generates a moodle assignment draft') !== false) {
        return true;
    }
    if (strpos($normalized, 'generates a moodle practice quiz draft') !== false) {
        return true;
    }

    return false;
}

/**
 * Clean chat answer markdown for readability (chat-only sanitizer).
 *
 * @param string $answer
 * @return string
 */
function local_chatbot_normalize_chat_answer(string $answer): string {
    $text = trim(str_replace(["\r\n", "\r"], "\n", $answer));
    if ($text === '') {
        return '';
    }
    $original = $text;

    $patterns = [
        '/^\s*#{1,6}\s*answer\b[\s:\-]*/iu',
        '/^\s*answer\s*:\s*(?=\*{1,2}\s*answer\s*\*{1,2}\s*:)/iu',
    ];
    for ($i = 0; $i < 6; $i++) {
        $changed = false;
        foreach ($patterns as $pattern) {
            $updated = preg_replace($pattern, '', $text, 1);
            if (is_string($updated) && $updated !== $text) {
                $text = ltrim($updated);
                $changed = true;
            }
        }
        if (!$changed) {
            break;
        }
    }

    $text = (string)preg_replace('/\s+---\s+/u', "\n\n", $text);
    $text = (string)preg_replace(
        '/\s+(\*{1,2}\s*(answer|reasoning|example|tip|advice|common mistake|challenge|practice question)\s*\*{1,2}\s*:)/iu',
        "\n\n$1",
        $text
    );
    // Convert inline dash bullets (e.g. "Tips: - A - B") into proper markdown list lines.
    // We trigger on sentence punctuation + " - " so prose stays intact.
    $text = (string)preg_replace('/([:;.!?])\s+-\s+/u', "$1\n\n- ", $text);
    for ($i = 0; $i < 12; $i++) {
        $updated = preg_replace('/(\n-\s+[^\n]*?)\s+-\s+/u', "$1\n- ", $text, 1);
        if (!is_string($updated) || $updated === $text) {
            break;
        }
        $text = $updated;
    }
    // Moodle markdown parser needs a blank line before list markers
    // to render bullet/numbered lists consistently.
    $lines = explode("\n", $text);
    $normalizedlines = [];
    foreach ($lines as $line) {
        $islistmarker = preg_match('/^(?:[-*+]|\d+\.)\s+/u', ltrim($line)) === 1;
        if ($islistmarker) {
            $prevline = end($normalizedlines);
            $prevtrim = $prevline === false ? '' : trim((string)$prevline);
            $previslist = $prevline !== false &&
                preg_match('/^(?:[-*+]|\d+\.)\s+/u', ltrim((string)$prevline)) === 1;
            if ($prevtrim !== '' && !$previslist) {
                $normalizedlines[] = '';
            }
        }
        $normalizedlines[] = $line;
    }
    $text = implode("\n", $normalizedlines);

    $blocks = preg_split('/\n\s*\n/u', $text) ?: [];
    $seen = [];
    $cleanblocks = [];
    foreach ($blocks as $block) {
        $block = trim((string)$block);
        if ($block === '') {
            continue;
        }
        $key = core_text::strtolower(trim((string)preg_replace('/\s+/u', ' ', $block)));
        if ($key === '' || isset($seen[$key])) {
            continue;
        }
        $seen[$key] = true;
        $cleanblocks[] = $block;
    }

    if (!empty($cleanblocks)) {
        $text = implode("\n\n", $cleanblocks);
    }
    $text = (string)preg_replace('/[ \t]+\n/u', "\n", $text);
    $text = (string)preg_replace('/\n{3,}/u', "\n\n", $text);
    $text = trim($text);

    return $text === '' ? $original : $text;
}

/**
 * Map mastery percent into low|mid|high.
 *
 * @param float|null $mastery
 * @param string $defaultgroup
 * @return string
 */
function local_chatbot_map_mastery_to_group(?float $mastery, string $defaultgroup = 'mid'): string {
    $allowed = ['low' => true, 'mid' => true, 'high' => true];
    $defaultgroup = trim(core_text::strtolower($defaultgroup));
    if (!isset($allowed[$defaultgroup])) {
        $defaultgroup = 'mid';
    }

    if ($mastery === null) {
        return $defaultgroup;
    }

    $mastery = local_chatbot_normalize_mastery_percent((float)$mastery);
    if ($mastery <= 69.0) {
        return 'low';
    }
    if ($mastery <= 84.0) {
        return 'mid';
    }
    return 'high';
}

/**
 * Normalize topic text into stable matching key.
 *
 * @param string $topic
 * @return string
 */
function local_chatbot_normalize_topic_key(string $topic): string {
    $topic = html_entity_decode($topic, ENT_QUOTES | ENT_HTML5, 'UTF-8');
    $topic = trim((string)preg_replace('/\s+/', ' ', $topic));
    if ($topic === '') {
        return '';
    }
    return core_text::strtolower($topic);
}

/**
 * Resolve canonical topic name from a requested topic.
 *
 * @param int $courseid
 * @param int $userid
 * @param string $requestedtopic
 * @return array{status:string,requested_topic:string,active_topic:?string}
 */
function local_chatbot_resolve_course_topic_name(int $courseid, int $userid, string $requestedtopic): array {
    $requestedtopic = trim($requestedtopic);
    if ($courseid <= 0 || $userid <= 0 || $requestedtopic === '') {
        return [
            'status' => 'topic_not_resolved',
            'requested_topic' => $requestedtopic,
            'active_topic' => null,
        ];
    }

    $requestedkey = local_chatbot_normalize_topic_key($requestedtopic);
    if ($requestedkey === '') {
        return [
            'status' => 'topic_not_resolved',
            'requested_topic' => $requestedtopic,
            'active_topic' => null,
        ];
    }

    $topics = local_chatbot_list_course_topics($courseid, $userid);
    foreach ($topics as $topicitem) {
        $candidate = trim((string)($topicitem['value'] ?? $topicitem['label'] ?? ''));
        if ($candidate === '') {
            continue;
        }
        if (local_chatbot_normalize_topic_key($candidate) === $requestedkey) {
            return [
                'status' => 'ok',
                'requested_topic' => $requestedtopic,
                'active_topic' => $candidate,
            ];
        }
    }

    return [
        'status' => 'topic_not_found_in_course',
        'requested_topic' => $requestedtopic,
        'active_topic' => null,
    ];
}

/**
 * Resolve active topic context and map to mastery group.
 *
 * @param int $userid
 * @param int $courseid
 * @param string $requestedtopic
 * @param string $defaultgroup
 * @return array{
 *   status:string,
 *   userid:int,
 *   courseid:int,
 *   requested_topic:string,
 *   active_topic:?string,
 *   selection_rule:string,
 *   source:string,
 *   mastery:?float,
 *   group:string,
 *   fallback_group:?string,
 *   attempt_count:int,
 *   last_event_time:?int,
 *   timemodified:?int
 * }
 */
function local_chatbot_resolve_active_topic_context(
    int $userid,
    int $courseid,
    string $requestedtopic,
    string $defaultgroup = 'mid'
): array {
    $topiccontext = local_chatbot_resolve_course_topic_name($courseid, $userid, $requestedtopic);
    $activetopic = $topiccontext['active_topic'];

    if ($activetopic === null) {
        $group = local_chatbot_map_mastery_to_group(null, $defaultgroup);
        return [
            'status' => (string)$topiccontext['status'],
            'userid' => $userid,
            'courseid' => $courseid,
            'requested_topic' => trim($requestedtopic),
            'active_topic' => null,
            'selection_rule' => 'fallback_default_group',
            'source' => 'local_chatbot_std_profile',
            'mastery' => null,
            'group' => $group,
            'fallback_group' => $group,
            'attempt_count' => 0,
            'last_event_time' => null,
            'timemodified' => null,
        ];
    }

    $masterymap = local_chatbot_get_user_topic_mastery_map($userid, $courseid);
    $mastery = array_key_exists($activetopic, $masterymap) ? (float)$masterymap[$activetopic] : null;
    $group = local_chatbot_map_mastery_to_group($mastery, $defaultgroup);

    return [
        'status' => $mastery === null ? 'no_topic_mastery_data' : 'ok',
        'userid' => $userid,
        'courseid' => $courseid,
        'requested_topic' => trim($requestedtopic),
        'active_topic' => $activetopic,
        'selection_rule' => 'single_topic',
        'source' => 'local_chatbot_std_profile',
        'mastery' => $mastery,
        'group' => $group,
        'fallback_group' => $mastery === null ? $group : null,
        'attempt_count' => 0,
        'last_event_time' => null,
        'timemodified' => null,
    ];
}

/**
 * Build task-generation difficulty modifier from mastery group.
 *
 * @param string $group
 * @param string $topic
 * @return string
 */
function local_chatbot_build_task_generation_level_modifier(string $group, string $topic = ''): string {
    $group = core_text::strtolower(trim($group));
    if (!in_array($group, ['low', 'mid', 'high'], true)) {
        $group = 'mid';
    }

    $topicline = '';
    $topic = trim($topic);
    if ($topic !== '') {
        $topicline = "Active topic: {$topic}\n";
    }

    $difficultyline = 'Difficulty target: standard (mid mastery baseline).';
    if ($group === 'low') {
        $difficultyline = 'Difficulty target: easier than standard (low mastery).';
    } else if ($group === 'high') {
        $difficultyline = 'Difficulty target: more challenging than standard (high mastery).';
    }

    return
        "Task generation mode: topic-mastery adaptive.\n" .
        $topicline .
        $difficultyline . "\n" .
        "Difficulty must change through reasoning depth and question framing, not by introducing unrelated topics.\n" .
        "Use only concepts explicitly supported by the selected PDF/topic context.\n" .
        "If reference material is limited, keep concept scope fixed and adjust complexity through phrasing and inference demand.\n";
}

/**
 * Build chatbot language-style modifier from mastery group.
 *
 * @param string $group
 * @param string $topic
 * @return string
 */
function local_chatbot_build_chatbot_language_level_modifier(string $group, string $topic = ''): string {
    $group = core_text::strtolower(trim($group));
    if (!in_array($group, ['low', 'mid', 'high'], true)) {
        $group = 'mid';
    }

    $topicline = '';
    $topic = trim($topic);
    if ($topic !== '') {
        $topicline = "Active topic: {$topic}\n";
    }

    if ($group === 'low') {
        return
            "Chat mode: low mastery language adaptation.\n" .
            $topicline .
            "Use very simple language, short sentences, and beginner-friendly wording.\n" .
            "Avoid jargon when possible; if unavoidable, define it in plain words.\n" .
            "Keep explanation concise and easy to scan (target about 80-140 words).\n" .
            "Prioritize clarity over completeness.\n";
    }

    if ($group === 'high') {
        return
            "Chat mode: high mastery language adaptation.\n" .
            $topicline .
            "Use more technical and precise language with deeper explanation.\n" .
            "Include stronger reasoning detail and richer conceptual linkage.\n" .
            "Answer can be longer and denser (target about 220-320 words).\n";
    }

    return
        "Chat mode: mid mastery language adaptation.\n" .
        $topicline .
        "Use clear language with moderate technical detail.\n" .
        "Give balanced explanation depth with concise reasoning.\n" .
        "Keep answer medium length (target about 140-220 words).\n";
}

/**
 * Build chatbot output instruction by mastery group.
 *
 * @param string $group
 * @param string $topic
 * @return string
 */
function local_chatbot_build_chatbot_level_modifier(string $group, string $topic = ''): string {
    $group = core_text::strtolower(trim($group));
    $topicline = '';
    $topic = trim($topic);
    if ($topic !== '') {
        $topicline = "Active topic: {$topic}\n";
    }

    if ($group === 'low') {
        return
            "Student output level: low mastery.\n" .
            $topicline .
            "Answer style requirements:\n" .
            "- Use this exact section order:\n" .
            "  **Answer:**\n" .
            "  **Example:**\n" .
            "- Keep total answer <= 160 words.\n" .
            "- Use simple language and short sentences.\n" .
            "- Do not include practice questions, challenge questions, or quiz tasks.\n";
    }
    if ($group === 'high') {
        return
            "Student output level: high mastery.\n" .
            $topicline .
            "Answer style requirements:\n" .
            "- Use this exact section order:\n" .
            "  **Answer:**\n" .
            "  **Reasoning:**\n" .
            "  **Comparison:**\n" .
            "  **Transfer Insight:**\n" .
            "- Keep total answer <= 260 words.\n" .
            "- Focus on reasoning and concept comparison.\n" .
            "- Do not include extension challenge questions.\n";
    }

    return
        "Student output level: mid mastery.\n" .
        $topicline .
        "Answer style requirements:\n" .
        "- Use this exact section order:\n" .
        "  **Answer:**\n" .
        "  **Reasoning:**\n" .
        "  **Example:**\n" .
        "- Keep total answer <= 220 words.\n" .
        "- Do not include practice questions or challenge questions.\n";
}

/**
 * Count words in a text (unicode-aware).
 *
 * @param string $text
 * @return int
 */
function local_chatbot_count_words(string $text): int {
    $matches = [];
    preg_match_all('/[\p{L}\p{N}]+/u', $text, $matches);
    return isset($matches[0]) ? count($matches[0]) : 0;
}

/**
 * Check whether one section label exists in answer text.
 *
 * @param string $text
 * @param string $label
 * @return bool
 */
function local_chatbot_has_chat_section_label(string $text, string $label): bool {
    $pattern = '/(?:^|\n)\s*(?:\*{1,2}\s*)?' . preg_quote($label, '/') . '(?:\s*\*{1,2})?\s*:/iu';
    return preg_match($pattern, $text) === 1;
}

/**
 * Count occurrences of one section label in answer text.
 *
 * @param string $text
 * @param string $label
 * @return int
 */
function local_chatbot_count_chat_section_label(string $text, string $label): int {
    $pattern = '/(?:^|\n)\s*(?:\*{1,2}\s*)?' . preg_quote($label, '/') . '(?:\s*\*{1,2})?\s*:/iu';
    $matches = [];
    preg_match_all($pattern, $text, $matches);
    return isset($matches[0]) ? count($matches[0]) : 0;
}

/**
 * Validate style compliance for chat answer by mastery group.
 *
 * @param string $answer
 * @param string $group
 * @return bool
 */
function local_chatbot_is_chat_style_compliant(string $answer, string $group): bool {
    $group = core_text::strtolower(trim($group));
    $text = trim($answer);
    if ($text === '') {
        return false;
    }

    $wordcount = local_chatbot_count_words($text);
    $lowtext = core_text::strtolower($text);

    if (strpos($lowtext, 'practice question') !== false ||
        strpos($lowtext, 'challenge question') !== false ||
        strpos($lowtext, 'quiz task') !== false) {
        return false;
    }

    if ($group === 'low') {
        if ($wordcount > 160) {
            return false;
        }
        return local_chatbot_count_chat_section_label($text, 'Answer') === 1 &&
            local_chatbot_count_chat_section_label($text, 'Example') === 1;
    }
    if ($group === 'high') {
        if ($wordcount > 260) {
            return false;
        }
        return local_chatbot_has_chat_section_label($text, 'Answer') &&
            local_chatbot_has_chat_section_label($text, 'Reasoning') &&
            local_chatbot_has_chat_section_label($text, 'Comparison') &&
            local_chatbot_has_chat_section_label($text, 'Transfer Insight');
    }

    if ($wordcount > 220) {
        return false;
    }
    return local_chatbot_has_chat_section_label($text, 'Answer') &&
        local_chatbot_has_chat_section_label($text, 'Reasoning') &&
        local_chatbot_has_chat_section_label($text, 'Example');
}

/**
 * Build one rewrite prompt to force chat answer style.
 *
 * @param string $answer
 * @param string $group
 * @return string
 */
function local_chatbot_build_chat_style_rewrite_prompt(string $answer, string $group): string {
    $group = core_text::strtolower(trim($group));

    $template = "Use sections: **Answer:** **Reasoning:** **Example:** (<=220 words).";
    if ($group === 'low') {
        $template = "Use sections: **Answer:** **Example:** (<=160 words). "
            . "Simple language only. No practice/challenge questions.";
    } else if ($group === 'high') {
        $template = "Use sections: **Answer:** **Reasoning:** **Comparison:** **Transfer Insight:** (<=260 words). "
            . "No challenge questions.";
    }

    return
        "Rewrite the following answer to match a strict chat style.\n" .
        "Output only the rewritten final answer in markdown.\n" .
        $template . "\n\n" .
        "Original answer:\n" .
        $answer;
}

/**
 * Trim text to max words.
 *
 * @param string $text
 * @param int $maxwords
 * @return string
 */
function local_chatbot_trim_to_max_words(string $text, int $maxwords): string {
    $text = trim((string)preg_replace('/\s+/u', ' ', $text));
    if ($text === '' || $maxwords <= 0) {
        return '';
    }

    preg_match_all('/[\p{L}\p{N}]+(?:[\'\-][\p{L}\p{N}]+)*/u', $text, $matches, PREG_OFFSET_CAPTURE);
    $words = $matches[0] ?? [];
    if (count($words) <= $maxwords) {
        return $text;
    }

    $cut = $words[$maxwords - 1][1] + core_text::strlen($words[$maxwords - 1][0]);
    return rtrim(core_text::substr($text, 0, $cut), " ,.;:\t\n\r\0\x0B") . '.';
}

/**
 * Extract section body by markdown-style section label.
 *
 * @param string $text
 * @param string $label
 * @param array<int,string> $nextlabels
 * @return string
 */
function local_chatbot_extract_chat_section_content(string $text, string $label, array $nextlabels): string {
    $pattern = '/(?:^|\n)\s*(?:\*{1,2}\s*)?' . preg_quote($label, '/') . '(?:\s*\*{1,2})?\s*:\s*/iu';
    if (!preg_match($pattern, $text, $match, PREG_OFFSET_CAPTURE)) {
        return '';
    }

    $start = (int)$match[0][1] + core_text::strlen((string)$match[0][0]);
    $tail = core_text::substr($text, $start);
    $end = core_text::strlen($tail);

    foreach ($nextlabels as $nextlabel) {
        $nextpattern = '/(?:^|\n)\s*(?:\*{1,2}\s*)?' . preg_quote($nextlabel, '/') . '(?:\s*\*{1,2})?\s*:\s*/iu';
        if (preg_match($nextpattern, $tail, $nextmatch, PREG_OFFSET_CAPTURE)) {
            $pos = (int)$nextmatch[0][1];
            if ($pos >= 0) {
                $end = min($end, $pos);
            }
        }
    }

    return trim(core_text::substr($tail, 0, $end));
}

/**
 * Remove repeated section label prefix from section body.
 *
 * @param string $text
 * @return string
 */
function local_chatbot_strip_section_prefix(string $text): string {
    $text = trim($text);
    $text = (string)preg_replace(
        '/^(?:\*{1,2}\s*)?(answer|example|suggestion|tip|reasoning|comparison|transfer insight)(?:\s*\*{1,2})?\s*:\s*/iu',
        '',
        $text
    );
    return trim($text);
}

/**
 * Clean markdown artifacts from one chat section body.
 *
 * @param string $text
 * @return string
 */
function local_chatbot_sanitize_chat_section_body(string $text): string {
    $text = trim((string)$text);
    if ($text === '') {
        return '';
    }

    $text = (string)preg_replace('/^\s*(?:\*+\s*)+/u', '', $text);
    $text = str_replace(['**', '__', '`'], '', $text);
    $text = (string)preg_replace('/\s+/u', ' ', $text);
    $text = trim($text, " \t\n\r\0\x0B-:;");
    return trim($text);
}

/**
 * Return first sentence-like chunk from text.
 *
 * @param string $text
 * @return string
 */
function local_chatbot_first_sentence(string $text): string {
    $text = trim((string)preg_replace('/\s+/u', ' ', strip_tags($text)));
    if ($text === '') {
        return '';
    }
    $parts = preg_split('/(?<=[.!?])\s+/u', $text) ?: [];
    return trim((string)($parts[0] ?? $text));
}

/**
 * Build deterministic low-style fallback answer.
 *
 * @param string $answer
 * @return string
 */
function local_chatbot_build_low_style_fallback(string $answer): string {
    $text = trim($answer);
    $answerbody = local_chatbot_extract_chat_section_content($text, 'Answer', ['Reasoning', 'Example', 'Suggestion', 'Tip', 'Comparison', 'Transfer Insight']);
    $examplebody = local_chatbot_extract_chat_section_content($text, 'Example', ['Suggestion', 'Tip', 'Reasoning', 'Comparison', 'Transfer Insight']);

    if ($answerbody === '') {
        $answerbody = local_chatbot_first_sentence($text);
    }
    if ($examplebody === '') {
        $examplebody = 'AI hiring should treat all candidates fairly.';
    }
    $answerbody = local_chatbot_sanitize_chat_section_body(local_chatbot_strip_section_prefix($answerbody));
    $examplebody = local_chatbot_sanitize_chat_section_body(local_chatbot_strip_section_prefix($examplebody));

    $answerbody = local_chatbot_trim_to_max_words($answerbody, 55);
    $examplebody = local_chatbot_trim_to_max_words($examplebody, 45);

    $fallback = "**Answer:** {$answerbody}\n\n**Example:** {$examplebody}";
    if (local_chatbot_count_words($fallback) > 160) {
        $answerbody = local_chatbot_trim_to_max_words($answerbody, 35);
        $fallback = "**Answer:** {$answerbody}\n\n**Example:** {$examplebody}";
    }
    return trim($fallback);
}

/**
 * Enforce style by rewriting once when needed.
 *
 * @param string $answer
 * @param string $group
 * @return string
 */
function local_chatbot_enforce_chat_style(string $answer, string $group): string {
    $answer = trim($answer);
    if ($answer === '') {
        return $answer;
    }

    if (local_chatbot_is_chat_style_compliant($answer, $group)) {
        return $answer;
    }

    try {
        $rewriteprompt = local_chatbot_build_chat_style_rewrite_prompt($answer, $group);
        $rewritten = local_chatbot_run_llm_general($rewriteprompt, false);
        $candidate = trim((string)($rewritten['answer'] ?? ''));
        if ($candidate !== '' && local_chatbot_is_chat_style_compliant($candidate, $group)) {
            return $candidate;
        }
    } catch (\Throwable $e) {
        // Keep original answer when rewrite fails.
    }

    if ($group === 'low') {
        $fallback = local_chatbot_build_low_style_fallback($answer);
        if (local_chatbot_is_chat_style_compliant($fallback, 'low')) {
            return $fallback;
        }
    }

    return $answer;
}

/**
 * Detects whether user has teacher-like role assignment in any context.
 *
 * @param int $userid
 * @return bool
 */
function local_chatbot_user_is_teacher_like(int $userid): bool {
    global $DB;

    if ($userid <= 0) {
        return false;
    }

    $sql = "SELECT 1
              FROM {role_assignments} ra
              JOIN {role} r ON r.id = ra.roleid
             WHERE ra.userid = :userid
               AND r.shortname IN ('editingteacher', 'teacher', 'manager')";
    return $DB->record_exists_sql($sql, ['userid' => $userid]);
}

/**
 * Check whether user can access course materials for chatbot context.
 *
 * @param int $courseid
 * @param int $userid
 * @return bool
 */
function local_chatbot_user_can_access_course_materials(int $courseid, int $userid): bool {
    if ($courseid <= 0 || $userid <= 0) {
        return false;
    }

    $context = context_course::instance($courseid, IGNORE_MISSING);
    if (!$context) {
        return false;
    }

    if (is_siteadmin($userid)) {
        return true;
    }

    if (is_enrolled($context, $userid, '', true)) {
        return true;
    }

    return has_capability('moodle/course:view', $context, $userid) ||
        has_capability('moodle/course:update', $context, $userid);
}

/**
 * Lists class topics (section names) for a course the user can access.
 *
 * @param int $courseid
 * @param int $userid
 * @return array
 */
function local_chatbot_list_course_topics(int $courseid, int $userid): array {
    global $DB;

    if ($courseid <= 0 || $userid <= 0) {
        return [];
    }

    if (!$DB->record_exists('course', ['id' => $courseid])) {
        return [];
    }

    if (!local_chatbot_user_can_access_course_materials($courseid, $userid)) {
        return [];
    }

    $sections = $DB->get_records(
        'course_sections',
        ['course' => $courseid],
        'section ASC',
        'id,section,name'
    );
    $topics = [];
    $seen = [];

    foreach ($sections as $section) {
        if ((int)$section->section <= 0) {
            continue;
        }
        $name = trim((string)$section->name);
        if ($name === '') {
            $name = 'Topic ' . (int)$section->section;
        }
        if ($name === '') {
            continue;
        }
        if (isset($seen[$name])) {
            continue;
        }
        $seen[$name] = true;
        $topics[] = [
            'value' => $name,
            'label' => $name,
        ];
    }

    return $topics;
}

/**
 * Lists PDF resource files available in a course the user can access.
 *
 * @param int $courseid
 * @param int $userid
 * @param string $topic
 * @return array
 */
function local_chatbot_list_course_pdfs(int $courseid, int $userid, string $topic = ''): array {
    global $DB;

    if ($courseid <= 0 || $userid <= 0) {
        return [];
    }

    if (!$DB->record_exists('course', ['id' => $courseid])) {
        return [];
    }

    if (!local_chatbot_user_can_access_course_materials($courseid, $userid)) {
        return [];
    }

    $sql = "SELECT cm.id AS cmid, r.name, cs.section AS sectionnum, cs.name AS sectionname
              FROM {course_modules} cm
              JOIN {modules} m ON m.id = cm.module AND m.name = :modname
              JOIN {resource} r ON r.id = cm.instance
              JOIN {course_sections} cs ON cs.id = cm.section
             WHERE cm.course = :courseid
               AND cm.deletioninprogress = 0
          ORDER BY cm.id ASC";
    $records = $DB->get_records_sql($sql, ['modname' => 'resource', 'courseid' => $courseid]);
    if (!$records) {
        return [];
    }

    $fs = get_file_storage();
    $pdfs = [];
    $seen = [];

    foreach ($records as $record) {
        $sectionname = trim((string)$record->sectionname);
        if ($sectionname === '') {
            $sectionname = 'Topic ' . (int)$record->sectionnum;
        }

        if ($topic !== '') {
            if (core_text::strtolower(trim($sectionname)) !== core_text::strtolower(trim($topic))) {
                continue;
            }
        }

        $cmcontext = context_module::instance((int)$record->cmid, IGNORE_MISSING);
        if (!$cmcontext) {
            continue;
        }
        if (!is_siteadmin($userid) && !has_capability('mod/resource:view', $cmcontext, $userid)) {
            continue;
        }
        $files = $fs->get_area_files(
            $cmcontext->id,
            'mod_resource',
            'content',
            0,
            'filename ASC',
            false
        );
        if (!$files) {
            continue;
        }

        foreach ($files as $file) {
            $filename = (string)$file->get_filename();
            if (strtolower(pathinfo($filename, PATHINFO_EXTENSION)) !== 'pdf') {
                continue;
            }
            $label = trim((string)$record->name) !== ''
                ? (string)$record->name . ' (' . $filename . ')'
                : $filename;
            $value = $filename;
            if (isset($seen[$value])) {
                continue;
            }
            $seen[$value] = true;
            $pdfs[] = [
                'value' => $value,
                'label' => $label,
                'topic' => $sectionname,
            ];
        }
    }

    return $pdfs;
}

/**
 * Sync course topic materials (PDF/TXT resources) into chatbot data directory.
 *
 * @param int $courseid
 * @param int $userid
 * @param string $topic
 * @return array
 */
function local_chatbot_sync_course_topic_materials_to_data(int $courseid, int $userid, string $topic = ''): array {
    global $DB;

    if ($courseid <= 0 || $userid <= 0) {
        return [];
    }
    if (!$DB->record_exists('course', ['id' => $courseid])) {
        return [];
    }

    if (!local_chatbot_user_can_access_course_materials($courseid, $userid)) {
        return [];
    }

    $sql = "SELECT cm.id AS cmid, r.name, cs.section AS sectionnum, cs.name AS sectionname
              FROM {course_modules} cm
              JOIN {modules} m ON m.id = cm.module AND m.name = :modname
              JOIN {resource} r ON r.id = cm.instance
              JOIN {course_sections} cs ON cs.id = cm.section
             WHERE cm.course = :courseid
               AND cm.deletioninprogress = 0
          ORDER BY cm.id ASC";
    $records = $DB->get_records_sql($sql, ['modname' => 'resource', 'courseid' => $courseid]);
    if (!$records) {
        local_chatbot_clear_data_dir_documents();
        return [];
    }

    local_chatbot_clear_data_dir_documents();
    local_chatbot_ensure_data_dir();
    $datadir = local_chatbot_get_data_path();
    $fs = get_file_storage();
    $usednames = [];
    $topicnormalized = core_text::strtolower(trim($topic));

    foreach ($records as $record) {
        $sectionname = trim((string)$record->sectionname);
        if ($sectionname === '') {
            $sectionname = 'Topic ' . (int)$record->sectionnum;
        }

        if ($topicnormalized !== '') {
            $sectionnormalized = core_text::strtolower(trim($sectionname));
            if ($sectionnormalized !== $topicnormalized) {
                continue;
            }
        }

        $cmcontext = context_module::instance((int)$record->cmid, IGNORE_MISSING);
        if (!$cmcontext) {
            continue;
        }
        if (!is_siteadmin($userid) && !has_capability('mod/resource:view', $cmcontext, $userid)) {
            continue;
        }

        $files = $fs->get_area_files(
            $cmcontext->id,
            'mod_resource',
            'content',
            0,
            'filename ASC',
            false
        );
        if (!$files) {
            continue;
        }

        foreach ($files as $file) {
            $filename = (string)$file->get_filename();
            $ext = strtolower(pathinfo($filename, PATHINFO_EXTENSION));
            if ($ext !== 'pdf' && $ext !== 'txt') {
                continue;
            }

            $targetname = local_chatbot_unique_data_filename($filename, $usednames);
            $targetpath = $datadir . DIRECTORY_SEPARATOR . $targetname;
            $content = $file->get_content();
            if ($content === false) {
                continue;
            }
            file_put_contents($targetpath, $content);
        }
    }

    return local_chatbot_list_uploaded_files();
}

/**
 * Resolve an accessible course id from course label/shortname for current user.
 *
 * @param string $coursename
 * @param int $userid
 * @return int
 */
function local_chatbot_resolve_courseid_for_teacher(string $coursename, int $userid): int {
    global $DB;

    $coursename = trim($coursename);
    if ($coursename === '' || $userid <= 0) {
        return 0;
    }

    $sql = "SELECT id, shortname, fullname
              FROM {course}
             WHERE " . $DB->sql_compare_text('fullname') . " = :fullname
                OR " . $DB->sql_compare_text('shortname') . " = :shortname
          ORDER BY id ASC";
    $candidates = $DB->get_records_sql($sql, [
        'fullname' => $coursename,
        'shortname' => $coursename,
    ]);
    if (!$candidates) {
        return 0;
    }

    foreach ($candidates as $course) {
        if (local_chatbot_user_can_access_course_materials((int)$course->id, $userid)) {
            return (int)$course->id;
        }
    }
    return 0;
}

/**
 * Check if learning analytics tables are available.
 *
 * @return bool
 */
function local_chatbot_learning_tables_ready(): bool {
    global $DB;

    $dbman = $DB->get_manager();
    return $dbman->table_exists(new xmldb_table('local_chatbot_std_profile')) &&
        $dbman->table_exists(new xmldb_table('local_chatbot_learn_events'));
}

/**
 * Default minimum mastery percent for one topic.
 *
 * @return float
 */
function local_chatbot_mastery_minimum_default(): float {
    return 70.0;
}

/**
 * Normalize one mastery percent value into 0..100.
 *
 * @param float $value
 * @return float
 */
function local_chatbot_normalize_mastery_percent(float $value): float {
    return (float)round(min(100.0, max(0.0, $value)), 2);
}

/**
 * Load course mastery policy from plugin config.
 *
 * @param int $courseid
 * @return array{defaultminimum:float,requireforexams:int,overrides:array<string,float>}
 */
function local_chatbot_get_course_mastery_policy(int $courseid): array {
    $defaults = [
        'defaultminimum' => local_chatbot_mastery_minimum_default(),
        'requireforexams' => 1,
        'overrides' => [],
    ];
    if ($courseid <= 0) {
        return $defaults;
    }

    $configkey = 'mastery_policy_' . $courseid;
    $raw = (string)get_config('local_chatbot', $configkey);
    if (trim($raw) === '') {
        return $defaults;
    }

    $decoded = json_decode($raw, true);
    if (!is_array($decoded)) {
        return $defaults;
    }

    $defaultminimum = local_chatbot_normalize_mastery_percent((float)($decoded['defaultminimum'] ?? $defaults['defaultminimum']));
    $requireforexams = !empty($decoded['requireforexams']) ? 1 : 0;

    $overrides = [];
    if (!empty($decoded['overrides']) && is_array($decoded['overrides'])) {
        foreach ($decoded['overrides'] as $topic => $minimum) {
            $topickey = trim((string)$topic);
            if ($topickey === '') {
                continue;
            }
            $overrides[$topickey] = local_chatbot_normalize_mastery_percent((float)$minimum);
        }
    }

    return [
        'defaultminimum' => $defaultminimum,
        'requireforexams' => $requireforexams,
        'overrides' => $overrides,
    ];
}

/**
 * Persist course mastery policy into plugin config.
 *
 * @param int $courseid
 * @param float $defaultminimum
 * @param bool $requireforexams
 * @param array<string,float|int|string> $overrides
 * @return void
 */
function local_chatbot_save_course_mastery_policy(
    int $courseid,
    float $defaultminimum,
    bool $requireforexams,
    array $overrides
): void {
    if ($courseid <= 0) {
        return;
    }

    $cleanoverrides = [];
    foreach ($overrides as $topic => $minimum) {
        $topickey = trim((string)$topic);
        if ($topickey === '') {
            continue;
        }
        $cleanoverrides[$topickey] = local_chatbot_normalize_mastery_percent((float)$minimum);
    }
    ksort($cleanoverrides);

    $payload = [
        'defaultminimum' => local_chatbot_normalize_mastery_percent($defaultminimum),
        'requireforexams' => $requireforexams ? 1 : 0,
        'overrides' => $cleanoverrides,
    ];
    $json = json_encode($payload);
    if ($json === false) {
        return;
    }

    set_config('mastery_policy_' . $courseid, $json, 'local_chatbot');
}

/**
 * Resolve minimum mastery for one topic based on course policy.
 *
 * @param string $topic
 * @param array{defaultminimum:float,requireforexams:int,overrides:array<string,float>} $policy
 * @return float
 */
function local_chatbot_get_course_topic_minimum(string $topic, array $policy): float {
    $topickey = trim($topic);
    if ($topickey !== '' && isset($policy['overrides'][$topickey])) {
        return local_chatbot_normalize_mastery_percent((float)$policy['overrides'][$topickey]);
    }
    return local_chatbot_normalize_mastery_percent((float)($policy['defaultminimum'] ?? local_chatbot_mastery_minimum_default()));
}

/**
 * Check if mastery meets minimum threshold.
 *
 * @param float $mastery
 * @param float $minimum
 * @return bool
 */
function local_chatbot_mastery_meets_minimum(float $mastery, float $minimum): bool {
    return local_chatbot_normalize_mastery_percent($mastery) + 0.00001 >= local_chatbot_normalize_mastery_percent($minimum);
}

/**
 * Resolve mapped weighting bucket type for one activity.
 *
 * @param int $courseid
 * @param int $cmid
 * @return string
 */
function local_chatbot_get_activity_weight_bucket_type(int $courseid, int $cmid): string {
    if ($courseid <= 0 || $cmid <= 0) {
        return '';
    }
    if (!class_exists('\\local_chatbot\\service\\weight_ui_service')) {
        return '';
    }
    if (!\local_chatbot\service\weight_ui_service::tables_ready()) {
        return '';
    }

    try {
        $scheme = \local_chatbot\service\weight_ui_service::get_or_create_active_scheme($courseid);
        $maps = \local_chatbot\service\weight_ui_service::get_activity_maps((int)$scheme->id);
        if (!isset($maps[$cmid])) {
            return '';
        }
        return trim((string)($maps[$cmid]->subtype ?? ''));
    } catch (\Throwable $e) {
        return '';
    }
}

/**
 * Resolve prior topic names before one activity section in the same course.
 *
 * @param int $courseid
 * @param int $cmid
 * @return array<int,string>
 */
function local_chatbot_get_prior_topic_names_for_activity(int $courseid, int $cmid): array {
    global $DB;

    if ($courseid <= 0 || $cmid <= 0) {
        return [];
    }

    $cm = $DB->get_record('course_modules', ['id' => $cmid, 'course' => $courseid], 'id,section', IGNORE_MISSING);
    if (!$cm || (int)$cm->section <= 0) {
        return [];
    }
    $examsection = $DB->get_record('course_sections', ['id' => (int)$cm->section], 'id,course,section', IGNORE_MISSING);
    if (!$examsection || (int)$examsection->section <= 0) {
        return [];
    }

    $sections = $DB->get_records_select(
        'course_sections',
        'course = :courseid AND section > 0 AND section < :sectionnum',
        ['courseid' => $courseid, 'sectionnum' => (int)$examsection->section],
        'section ASC',
        'id,section,name'
    );
    if (empty($sections)) {
        return [];
    }

    $topics = [];
    foreach ($sections as $section) {
        $name = trim((string)$section->name);
        if ($name === '') {
            $name = 'Topic ' . (int)$section->section;
        }
        if ($name === '') {
            continue;
        }
        $topics[$name] = $name;
    }
    return array_values($topics);
}

/**
 * Get mastery map keyed by topic for one user in one course.
 *
 * @param int $userid
 * @param int $courseid
 * @return array<string,float>
 */
function local_chatbot_get_user_topic_mastery_map(int $userid, int $courseid): array {
    global $DB;

    if ($userid <= 0 || $courseid <= 0 || !local_chatbot_learning_tables_ready()) {
        return [];
    }

    $records = $DB->get_records(
        'local_chatbot_std_profile',
        ['userid' => $userid, 'courseid' => $courseid],
        '',
        'topic,mastery'
    );
    $map = [];
    foreach ($records as $record) {
        $topic = trim((string)$record->topic);
        if ($topic === '') {
            continue;
        }
        $map[$topic] = local_chatbot_normalize_mastery_percent((float)$record->mastery);
    }
    return $map;
}

/**
 * Build debt rows for selected topics using current policy.
 *
 * @param int $userid
 * @param int $courseid
 * @param array<int,string> $topics
 * @param array{defaultminimum:float,requireforexams:int,overrides:array<string,float>}|null $policy
 * @return array<int,\stdClass>
 */
function local_chatbot_get_user_topic_debt_rows(
    int $userid,
    int $courseid,
    array $topics,
    ?array $policy = null
): array {
    if ($userid <= 0 || $courseid <= 0 || empty($topics)) {
        return [];
    }

    if ($policy === null) {
        $policy = local_chatbot_get_course_mastery_policy($courseid);
    }
    $masterymap = local_chatbot_get_user_topic_mastery_map($userid, $courseid);
    $rows = [];

    foreach ($topics as $topicname) {
        $topicname = trim((string)$topicname);
        if ($topicname === '') {
            continue;
        }
        $minimum = local_chatbot_get_course_topic_minimum($topicname, $policy);
        $mastery = (float)($masterymap[$topicname] ?? 0.0);
        $passed = local_chatbot_mastery_meets_minimum($mastery, $minimum);
        $rows[] = (object)[
            'topic' => $topicname,
            'mastery' => local_chatbot_normalize_mastery_percent($mastery),
            'minimum' => local_chatbot_normalize_mastery_percent($minimum),
            'passed' => $passed ? 1 : 0,
        ];
    }

    return $rows;
}

/**
 * Build student mastery status rows for one course topic list.
 *
 * @param int $userid
 * @param int $courseid
 * @return array<int,\stdClass>
 */
function local_chatbot_get_student_course_topic_mastery_status_rows(int $userid, int $courseid): array {
    global $DB;

    if ($userid <= 0 || $courseid <= 0) {
        return [];
    }

    $sections = $DB->get_records_select(
        'course_sections',
        'course = :courseid AND section > 0',
        ['courseid' => $courseid],
        'section ASC',
        'id,section,name'
    );
    $topics = [];
    $seen = [];
    foreach ($sections as $section) {
        $name = trim((string)$section->name);
        if ($name === '') {
            $name = 'Topic ' . (int)$section->section;
        }
        if ($name === '' || isset($seen[$name])) {
            continue;
        }
        $seen[$name] = true;
        $topics[] = ['value' => $name];
    }
    if (empty($topics)) {
        return [];
    }

    $policy = local_chatbot_get_course_mastery_policy($courseid);
    $masterymap = [];
    if (local_chatbot_learning_tables_ready()) {
        $profiles = local_chatbot_get_student_mastery_rows($userid);
        foreach ($profiles as $profile) {
            if ((int)($profile->courseid ?? 0) !== $courseid) {
                continue;
            }
            $topic = trim((string)($profile->topic ?? ''));
            if ($topic === '') {
                continue;
            }
            $masterymap[$topic] = local_chatbot_normalize_mastery_percent((float)($profile->mastery ?? 0.0));
        }
    }

    $rows = [];
    foreach ($topics as $topicitem) {
        $topic = trim((string)($topicitem['value'] ?? ''));
        if ($topic === '') {
            continue;
        }

        $hasdata = array_key_exists($topic, $masterymap);
        $mastery = $hasdata ? (float)$masterymap[$topic] : 0.0;
        $minimum = local_chatbot_get_course_topic_minimum($topic, $policy);
        $passed = local_chatbot_mastery_meets_minimum($mastery, $minimum);

        $rows[] = (object)[
            'topic' => $topic,
            'mastery' => local_chatbot_normalize_mastery_percent($mastery),
            'minimum' => local_chatbot_normalize_mastery_percent($minimum),
            'passed' => $passed ? 1 : 0,
            'hasdata' => $hasdata ? 1 : 0,
        ];
    }

    return $rows;
}

/**
 * Get mastery rows for one student.
 *
 * @param int $userid
 * @return array
 */
function local_chatbot_get_student_mastery_rows(int $userid): array {
    global $DB;

    if ($userid <= 0 || !local_chatbot_learning_tables_ready()) {
        return [];
    }

    $sql = "SELECT p.*, c.fullname, c.shortname
              FROM {local_chatbot_std_profile} p
              JOIN {course} c ON c.id = p.courseid
             WHERE p.userid = :userid
          ORDER BY p.timemodified DESC, p.mastery DESC";
    return array_values($DB->get_records_sql($sql, ['userid' => $userid]));
}

/**
 * Get class-level mastery aggregates for one student.
 *
 * @param int $userid
 * @return array
 */
function local_chatbot_get_student_class_mastery_rows(int $userid): array {
    global $DB;

    if ($userid <= 0 || !local_chatbot_learning_tables_ready()) {
        return [];
    }

    $sql = "SELECT p.courseid, c.fullname, c.shortname,
                   COUNT(1) AS topiccount,
                   SUM(p.attempt_count) AS attemptsum,
                   CASE WHEN SUM(p.attempt_count) > 0
                        THEN SUM(p.mastery * p.attempt_count) / SUM(p.attempt_count)
                        ELSE AVG(p.mastery)
                   END AS classmastery,
                   CASE WHEN SUM(p.attempt_count) > 0
                        THEN SUM(p.accuracy_avg * p.attempt_count) / SUM(p.attempt_count)
                        ELSE AVG(p.accuracy_avg)
                   END AS classaccuracy,
                   MAX(p.timemodified) AS lastupdate
              FROM {local_chatbot_std_profile} p
              JOIN {course} c ON c.id = p.courseid
             WHERE p.userid = :userid
          GROUP BY p.courseid, c.fullname, c.shortname
          ORDER BY classmastery DESC, lastupdate DESC, c.fullname ASC";

    return array_values($DB->get_records_sql($sql, ['userid' => $userid]));
}

/**
 * Get overall mastery aggregates for one student.
 *
 * @param int $userid
 * @return array{overallmastery:float,overallaccuracy:float,classcount:int,topiccount:int,attemptsum:int,lastupdate:int}
 */
function local_chatbot_get_student_overall_mastery(int $userid): array {
    global $DB;

    $defaults = [
        'overallmastery' => 0.0,
        'overallaccuracy' => 0.0,
        'classcount' => 0,
        'topiccount' => 0,
        'attemptsum' => 0,
        'lastupdate' => 0,
    ];

    if ($userid <= 0 || !local_chatbot_learning_tables_ready()) {
        return $defaults;
    }

    $sql = "SELECT COUNT(DISTINCT p.courseid) AS classcount,
                   COUNT(1) AS topiccount,
                   SUM(p.attempt_count) AS attemptsum,
                   MAX(p.timemodified) AS lastupdate,
                   CASE WHEN SUM(p.attempt_count) > 0
                        THEN SUM(p.mastery * p.attempt_count) / SUM(p.attempt_count)
                        ELSE AVG(p.mastery)
                   END AS overallmastery,
                   CASE WHEN SUM(p.attempt_count) > 0
                        THEN SUM(p.accuracy_avg * p.attempt_count) / SUM(p.attempt_count)
                        ELSE AVG(p.accuracy_avg)
                   END AS overallaccuracy
              FROM {local_chatbot_std_profile} p
             WHERE p.userid = :userid";
    $record = $DB->get_record_sql($sql, ['userid' => $userid], IGNORE_MISSING);
    if (!$record) {
        return $defaults;
    }

    return [
        'overallmastery' => isset($record->overallmastery) ? (float)$record->overallmastery : 0.0,
        'overallaccuracy' => isset($record->overallaccuracy) ? (float)$record->overallaccuracy : 0.0,
        'classcount' => isset($record->classcount) ? (int)$record->classcount : 0,
        'topiccount' => isset($record->topiccount) ? (int)$record->topiccount : 0,
        'attemptsum' => isset($record->attemptsum) ? (int)$record->attemptsum : 0,
        'lastupdate' => isset($record->lastupdate) ? (int)$record->lastupdate : 0,
    ];
}

/**
 * Build teacher-facing mastery dashboard dataset.
 *
 * @param array $courseids
 * @return array
 */
function local_chatbot_get_teacher_mastery_dashboard(array $courseids): array {
    global $DB;

    $dataset = [
        'summary' => [
            'studentcount' => 0,
            'profilecount' => 0,
            'avgmastery' => 0.0,
            'eventcount' => 0,
            'lastupdate' => 0,
        ],
        'topics' => [],
        'learners' => [],
        'events' => [],
    ];

    if (!local_chatbot_learning_tables_ready()) {
        return $dataset;
    }

    $normalizedids = [];
    foreach ($courseids as $courseid) {
        $id = (int)$courseid;
        if ($id > 0) {
            $normalizedids[$id] = $id;
        }
    }
    if (empty($normalizedids)) {
        return $dataset;
    }

    [$insql, $params] = $DB->get_in_or_equal(array_values($normalizedids), SQL_PARAMS_NAMED, 'c');

    $dataset['summary']['studentcount'] = (int)$DB->count_records_sql(
        "SELECT COUNT(DISTINCT p.userid)
           FROM {local_chatbot_std_profile} p
          WHERE p.courseid {$insql}",
        $params
    );
    $dataset['summary']['profilecount'] = (int)$DB->count_records_sql(
        "SELECT COUNT(1)
           FROM {local_chatbot_std_profile} p
          WHERE p.courseid {$insql}",
        $params
    );

    $avgmastery = $DB->get_field_sql(
        "SELECT AVG(p.mastery)
           FROM {local_chatbot_std_profile} p
          WHERE p.courseid {$insql}",
        $params
    );
    $dataset['summary']['avgmastery'] = $avgmastery !== false ? (float)$avgmastery : 0.0;

    $dataset['summary']['eventcount'] = (int)$DB->count_records_sql(
        "SELECT COUNT(1)
           FROM {local_chatbot_learn_events} e
          WHERE e.courseid {$insql}",
        $params
    );

    $lastprofileupdate = (int)$DB->get_field_sql(
        "SELECT MAX(p.timemodified)
           FROM {local_chatbot_std_profile} p
          WHERE p.courseid {$insql}",
        $params
    );
    $lasteventupdate = (int)$DB->get_field_sql(
        "SELECT MAX(e.timecreated)
           FROM {local_chatbot_learn_events} e
          WHERE e.courseid {$insql}",
        $params
    );
    $dataset['summary']['lastupdate'] = max($lastprofileupdate, $lasteventupdate);

    $dataset['topics'] = array_values($DB->get_records_sql(
        "SELECT CONCAT(p.courseid, ':', p.topic) AS rowid,
                p.courseid, c.fullname, c.shortname, p.topic,
                AVG(p.mastery) AS avgmastery,
                AVG(p.accuracy_avg) AS avgaccuracy,
                COUNT(DISTINCT p.userid) AS learnercount,
                SUM(p.attempt_count) AS attemptsum,
                MAX(p.timemodified) AS lastupdate
           FROM {local_chatbot_std_profile} p
           JOIN {course} c ON c.id = p.courseid
          WHERE p.courseid {$insql}
       GROUP BY p.courseid, c.fullname, c.shortname, p.topic
       ORDER BY avgmastery ASC, attemptsum DESC, p.topic ASC",
        $params,
        0,
        100
    ));

    $dataset['learners'] = array_values($DB->get_records_sql(
        "SELECT CONCAT(p.courseid, ':', p.userid) AS rowid,
                p.courseid, c.fullname, c.shortname, p.userid,
                u.firstname, u.lastname,
                AVG(p.mastery) AS avgmastery,
                AVG(p.accuracy_avg) AS avgaccuracy,
                SUM(p.attempt_count) AS attemptsum,
                MAX(p.timemodified) AS lastupdate
           FROM {local_chatbot_std_profile} p
           JOIN {course} c ON c.id = p.courseid
           JOIN {user} u ON u.id = p.userid
          WHERE p.courseid {$insql}
       GROUP BY p.courseid, c.fullname, c.shortname, p.userid, u.firstname, u.lastname
       ORDER BY avgmastery ASC, attemptsum DESC, u.firstname ASC, u.lastname ASC",
        $params,
        0,
        100
    ));

    $dataset['events'] = array_values($DB->get_records_sql(
        "SELECT e.id AS rowid,
                e.courseid, c.fullname, c.shortname, e.userid,
                u.firstname, u.lastname,
                e.topic, e.event_type, e.score_topic, e.duration_seconds, e.submitted_at, e.timecreated
           FROM {local_chatbot_learn_events} e
           JOIN {course} c ON c.id = e.courseid
           JOIN {user} u ON u.id = e.userid
          WHERE e.courseid {$insql}
       ORDER BY e.timecreated DESC",
        $params,
        0,
        100
    ));

    return $dataset;
}

/**
 * Check whether weekly snapshot table is available.
 *
 * @return bool
 */
function local_chatbot_weekly_snapshot_ready(): bool {
    global $DB;
    return $DB->get_manager()->table_exists(new xmldb_table('local_chatbot_weekly_snap'));
}

/**
 * Get topic-level progress metrics for one student.
 *
 * @param int $userid
 * @param float $targetmastery mastery target in percent (e.g. 75 for 0.75)
 * @return array
 */
function local_chatbot_get_student_topic_progress_rows(int $userid, float $targetmastery = 75.0): array {
    global $DB;

    if ($userid <= 0 || !local_chatbot_learning_tables_ready()) {
        return [];
    }

    $profiles = local_chatbot_get_student_mastery_rows($userid);
    if (empty($profiles)) {
        return [];
    }

    $firstattemptmap = [];
    $eventrows = $DB->get_records_sql(
        "SELECT id, courseid, topic, score_topic, submitted_at
           FROM {local_chatbot_learn_events}
          WHERE userid = :userid
       ORDER BY submitted_at ASC, id ASC",
        ['userid' => $userid]
    );
    foreach ($eventrows as $event) {
        $key = (int)$event->courseid . '|' . (string)$event->topic;
        if (!isset($firstattemptmap[$key])) {
            $firstattemptmap[$key] = [
                'score_topic' => (float)$event->score_topic,
                'submitted_at' => (int)$event->submitted_at,
            ];
        }
    }

    $dailymap = [];
    foreach ($eventrows as $event) {
        $key = (int)$event->courseid . '|' . (string)$event->topic;
        $daystart = local_chatbot_get_day_start_utc((int)$event->submitted_at);
        if (!isset($dailymap[$key])) {
            $dailymap[$key] = [];
        }
        if (!isset($dailymap[$key][$daystart])) {
            $dailymap[$key][$daystart] = ['sum' => 0.0, 'count' => 0];
        }
        $dailymap[$key][$daystart]['sum'] += (float)$event->score_topic;
        $dailymap[$key][$daystart]['count']++;
    }

    $rows = [];
    foreach ($profiles as $profile) {
        $key = (int)$profile->courseid . '|' . (string)$profile->topic;
        $daily = $dailymap[$key] ?? [];
        $firstattempt = $firstattemptmap[$key] ?? null;

        ksort($daily);
        $trendpoints = [];
        foreach ($daily as $bucket) {
            $count = max(1, (int)$bucket['count']);
            $trendpoints[] = (float)$bucket['sum'] / $count;
        }
        if (empty($trendpoints)) {
            $trendpoints[] = (float)$profile->mastery;
        }

        $firstmastery = (float)reset($trendpoints);
        $lastmastery = (float)end($trendpoints);
        $masterychange = $lastmastery - $firstmastery;

        $starttime = $firstattempt ? (int)$firstattempt['submitted_at'] : 0;
        if ($starttime <= 0 && !empty($daily)) {
            $starttime = (int)array_key_first($daily);
        }

        $reachedtime = 0;
        foreach ($daily as $daystart => $bucket) {
            $avg = (float)$bucket['sum'] / max(1, (int)$bucket['count']);
            if ($avg >= $targetmastery) {
                $reachedtime = (int)$daystart;
                break;
            }
        }
        if ($reachedtime <= 0 && (float)$profile->mastery >= $targetmastery && (int)$profile->last_event_time > 0) {
            $reachedtime = (int)$profile->last_event_time;
        }

        $timetotarget = null;
        if ($starttime > 0 && $reachedtime > 0 && $reachedtime >= $starttime) {
            $timetotarget = $reachedtime - $starttime;
        }

        $row = clone $profile;
        $row->mastery_change = $masterychange;
        $row->first_attempt_accuracy = $firstattempt ? (float)$firstattempt['score_topic'] : null;
        $row->time_to_target_seconds = $timetotarget;
        $row->target_reached = $reachedtime > 0;
        $row->target_mastery = $targetmastery;
        $row->trend_points = array_slice($trendpoints, -14);
        $rows[] = $row;
    }

    usort($rows, static function($a, $b): int {
        $masterycmp = (float)$a->mastery <=> (float)$b->mastery;
        if ($masterycmp !== 0) {
            return $masterycmp;
        }
        return strcmp((string)$a->topic, (string)$b->topic);
    });

    return $rows;
}

/**
 * Get teacher topic-level progress metrics across selected courses.
 *
 * @param array $courseids
 * @param float $targetmastery mastery target in percent (e.g. 75 for 0.75)
 * @param int $limit
 * @return array
 */
function local_chatbot_get_teacher_topic_progress_rows(
    array $courseids,
    float $targetmastery = 75.0,
    int $limit = 200
): array {
    global $DB;

    if (!local_chatbot_learning_tables_ready()) {
        return [];
    }

    $normalizedids = [];
    foreach ($courseids as $courseid) {
        $id = (int)$courseid;
        if ($id > 0) {
            $normalizedids[$id] = $id;
        }
    }
    if (empty($normalizedids)) {
        return [];
    }

    [$insql, $params] = $DB->get_in_or_equal(array_values($normalizedids), SQL_PARAMS_NAMED, 'tc');
    $profiles = array_values($DB->get_records_sql(
        "SELECT CONCAT(p.courseid, ':', p.userid, ':', p.topic) AS rowid,
                p.userid, p.courseid, p.topic, p.mastery, p.accuracy_avg, p.attempt_count, p.last_event_time, p.timemodified,
                u.firstname, u.lastname,
                c.fullname, c.shortname
           FROM {local_chatbot_std_profile} p
           JOIN {user} u ON u.id = p.userid
           JOIN {course} c ON c.id = p.courseid
          WHERE p.courseid {$insql}
       ORDER BY p.mastery ASC, p.timemodified DESC",
        $params
    ));
    if (empty($profiles)) {
        return [];
    }

    $firstattemptmap = [];
    $eventrows = $DB->get_records_sql(
        "SELECT id, userid, courseid, topic, score_topic, submitted_at
           FROM {local_chatbot_learn_events}
          WHERE courseid {$insql}
       ORDER BY submitted_at ASC, id ASC",
        $params
    );
    foreach ($eventrows as $event) {
        $key = (int)$event->userid . '|' . (int)$event->courseid . '|' . (string)$event->topic;
        if (!isset($firstattemptmap[$key])) {
            $firstattemptmap[$key] = [
                'score_topic' => (float)$event->score_topic,
                'submitted_at' => (int)$event->submitted_at,
            ];
        }
    }

    $dailymap = [];
    foreach ($eventrows as $event) {
        $key = (int)$event->userid . '|' . (int)$event->courseid . '|' . (string)$event->topic;
        $daystart = local_chatbot_get_day_start_utc((int)$event->submitted_at);
        if (!isset($dailymap[$key])) {
            $dailymap[$key] = [];
        }
        if (!isset($dailymap[$key][$daystart])) {
            $dailymap[$key][$daystart] = ['sum' => 0.0, 'count' => 0];
        }
        $dailymap[$key][$daystart]['sum'] += (float)$event->score_topic;
        $dailymap[$key][$daystart]['count']++;
    }

    $rows = [];
    foreach ($profiles as $profile) {
        $key = (int)$profile->userid . '|' . (int)$profile->courseid . '|' . (string)$profile->topic;
        $daily = $dailymap[$key] ?? [];
        $firstattempt = $firstattemptmap[$key] ?? null;

        ksort($daily);
        $trendpoints = [];
        foreach ($daily as $bucket) {
            $count = max(1, (int)$bucket['count']);
            $trendpoints[] = (float)$bucket['sum'] / $count;
        }
        if (empty($trendpoints)) {
            $trendpoints[] = (float)$profile->mastery;
        }

        $firstmastery = (float)reset($trendpoints);
        $lastmastery = (float)end($trendpoints);
        $masterychange = $lastmastery - $firstmastery;

        $starttime = $firstattempt ? (int)$firstattempt['submitted_at'] : 0;
        if ($starttime <= 0 && !empty($daily)) {
            $starttime = (int)array_key_first($daily);
        }

        $reachedtime = 0;
        foreach ($daily as $daystart => $bucket) {
            $avg = (float)$bucket['sum'] / max(1, (int)$bucket['count']);
            if ($avg >= $targetmastery) {
                $reachedtime = (int)$daystart;
                break;
            }
        }
        if ($reachedtime <= 0 && (float)$profile->mastery >= $targetmastery && (int)$profile->last_event_time > 0) {
            $reachedtime = (int)$profile->last_event_time;
        }

        $timetotarget = null;
        if ($starttime > 0 && $reachedtime > 0 && $reachedtime >= $starttime) {
            $timetotarget = $reachedtime - $starttime;
        }

        $row = clone $profile;
        $row->mastery_change = $masterychange;
        $row->first_attempt_accuracy = $firstattempt ? (float)$firstattempt['score_topic'] : null;
        $row->time_to_target_seconds = $timetotarget;
        $row->target_reached = $reachedtime > 0;
        $row->target_mastery = $targetmastery;
        $row->trend_points = array_slice($trendpoints, -14);
        $rows[] = $row;
    }

    usort($rows, static function($a, $b): int {
        $masterycmp = (float)$a->mastery <=> (float)$b->mastery;
        if ($masterycmp !== 0) {
            return $masterycmp;
        }
        return strcmp((string)$a->lastname . ' ' . (string)$a->firstname, (string)$b->lastname . ' ' . (string)$b->firstname);
    });

    return array_slice($rows, 0, max(1, $limit));
}

/**
 * Get day start (00:00:00 UTC) for a timestamp.
 *
 * @param int $timestamp
 * @return int
 */
function local_chatbot_get_day_start_utc(int $timestamp): int {
    $base = $timestamp > 0 ? $timestamp : time();
    $dt = new DateTime('@' . $base);
    $dt->setTimezone(new DateTimeZone('UTC'));
    $dt->setTime(0, 0, 0);
    return (int)$dt->getTimestamp();
}

/**
 * Render compact trend chart bars using daily points.
 *
 * @param array $points
 * @return string
 */
function local_chatbot_render_snapshot_trend_chart(array $points): string {
    if (empty($points)) {
        return '-';
    }

    $bars = [];
    foreach ($points as $point) {
        $value = min(100.0, max(0.0, (float)$point));
        $height = 4 + (int)round(($value / 100.0) * 16);
        $bars[] = html_writer::tag('span', '', [
            'style' => 'display:inline-block;width:6px;height:' . $height .
                'px;background:#0f6cbf;margin-right:2px;vertical-align:bottom;border-radius:2px;',
            'title' => format_float($value, 1) . '%',
        ]);
    }

    return html_writer::tag('span', implode('', $bars), [
        'style' => 'display:inline-block;height:22px;white-space:nowrap;',
    ]);
}

/**
 * Format seconds into compact duration text.
 *
 * @param int|null $seconds
 * @return string
 */
function local_chatbot_format_duration_short(?int $seconds): string {
    if ($seconds === null || $seconds < 0) {
        return '-';
    }
    if ($seconds < 3600) {
        return max(1, (int)round($seconds / 60)) . 'm';
    }
    if ($seconds < 86400) {
        $hours = (int)floor($seconds / 3600);
        $mins = (int)floor(($seconds % 3600) / 60);
        return $hours . 'h ' . $mins . 'm';
    }
    $days = (int)floor($seconds / 86400);
    $hours = (int)floor(($seconds % 86400) / 3600);
    return $days . 'd ' . $hours . 'h';
}
