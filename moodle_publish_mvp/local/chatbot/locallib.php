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
 * Generate trace request id.
 *
 * @return string
 */
function local_chatbot_generate_request_id(): string {
    try {
        $random = bin2hex(random_bytes(6));
    } catch (Throwable $e) {
        $random = substr(sha1(uniqid('', true)), 0, 12);
    }
    return 'req-' . gmdate('YmdHis') . '-' . $random;
}

/**
 * Returns path for end-to-end PHP trace log file.
 *
 * @return string
 */
function local_chatbot_get_trace_log_path(): string {
    global $CFG;
    $dir = $CFG->dataroot . DIRECTORY_SEPARATOR . 'local_chatbot' . DIRECTORY_SEPARATOR . 'logs';
    if (!is_dir($dir)) {
        @mkdir($dir, 0777, true);
    }
    return $dir . DIRECTORY_SEPARATOR . 'e2e_trace_php.jsonl';
}

/**
 * Returns path for python trace log file.
 *
 * @return string
 */
function local_chatbot_get_python_trace_log_path(): string {
    global $CFG;
    $dir = $CFG->dataroot . DIRECTORY_SEPARATOR . 'local_chatbot' . DIRECTORY_SEPARATOR . 'logs';
    if (!is_dir($dir)) {
        @mkdir($dir, 0777, true);
    }
    return $dir . DIRECTORY_SEPARATOR . 'e2e_trace_python.jsonl';
}

/**
 * Append structured trace event to JSONL file.
 *
 * @param string $event
 * @param array $context
 * @param string $level
 * @return void
 */
function local_chatbot_trace_log(string $event, array $context = [], string $level = 'info'): void {
    try {
        $payload = [
            'timestamp' => gmdate('c'),
            'ts_ms' => (int)round(microtime(true) * 1000),
            'layer' => 'php',
            'event' => $event,
            'level' => $level,
        ];
        foreach ($context as $key => $value) {
            $payload[(string)$key] = $value;
        }
        $line = json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_INVALID_UTF8_SUBSTITUTE);
        if ($line === false) {
            return;
        }
        @file_put_contents(local_chatbot_get_trace_log_path(), $line . PHP_EOL, FILE_APPEND | LOCK_EX);
    } catch (Throwable $e) {
        // Never break request flow because of logging failures.
    }
}

/**
 * Truncate long trace text to keep log files readable.
 *
 * @param string $text
 * @param int $maxchars
 * @return array{text:string,truncated:bool,length:int}
 */
function local_chatbot_trace_truncate_text(string $text, int $maxchars = 4000): array {
    $maxchars = max(200, $maxchars);
    $normalized = str_replace(["\r\n", "\r"], "\n", (string)$text);
    $length = core_text::strlen($normalized);
    if ($length <= $maxchars) {
        return [
            'text' => $normalized,
            'truncated' => false,
            'length' => $length,
        ];
    }
    $cut = core_text::substr($normalized, 0, $maxchars);
    return [
        'text' => $cut . '...(truncated)',
        'truncated' => true,
        'length' => $length,
    ];
}

/**
 * Normalize optional user page range.
 *
 * @param int $pagestart
 * @param int $pageend
 * @return array{page_start:int,page_end:int}
 */
function local_chatbot_normalize_page_range(int $pagestart, int $pageend): array {
    $start = max(0, $pagestart);
    $end = max(0, $pageend);

    if ($start > 0 && $end <= 0) {
        $end = $start;
    } else if ($end > 0 && $start <= 0) {
        $start = $end;
    }

    if ($start > 0 && $end > 0 && $end < $start) {
        $tmp = $start;
        $start = $end;
        $end = $tmp;
    }

    return [
        'page_start' => $start,
        'page_end' => $end,
    ];
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
 * Build document signature for current chatbot data directory files.
 *
 * @return array{signature:string,sources:int}
 */
function local_chatbot_get_data_dir_document_signature(): array {
    local_chatbot_ensure_data_dir();
    $datadir = local_chatbot_get_data_path();
    $parts = [];
    $sources = 0;

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
        $size = @filesize($path);
        $mtime = @filemtime($path);
        $parts[] = $name . ':' . (int)$size . ':' . (int)$mtime;
        $sources++;
    }

    sort($parts, SORT_STRING);
    return [
        'signature' => implode('|', $parts),
        'sources' => $sources,
    ];
}

/**
 * Read persisted parser manifest from data directory.
 *
 * @return array
 */
function local_chatbot_get_data_dir_parse_manifest(): array {
    $datadir = local_chatbot_get_data_path();
    $manifestpath = $datadir . DIRECTORY_SEPARATOR . '.rag_index_manifest.json';
    if (!is_file($manifestpath)) {
        return [];
    }
    $raw = @file_get_contents($manifestpath);
    if ($raw === false || trim($raw) === '') {
        return [];
    }
    $decoded = json_decode($raw, true);
    return is_array($decoded) ? $decoded : [];
}

/**
 * Get parse/index status for current synced data directory context.
 *
 * @return array{status:string,is_parsed:bool,sources:int,parsed_at:?int,signature:string,manifest_signature:string}
 */
function local_chatbot_get_current_material_parse_status(): array {
    $signatureinfo = local_chatbot_get_data_dir_document_signature();
    $manifest = local_chatbot_get_data_dir_parse_manifest();

    $signature = (string)($signatureinfo['signature'] ?? '');
    $sources = (int)($signatureinfo['sources'] ?? 0);
    $manifestsignature = trim((string)($manifest['signature'] ?? ''));
    $parsedat = isset($manifest['updated_at']) ? (int)$manifest['updated_at'] : null;
    if ($parsedat !== null && $parsedat <= 0) {
        $parsedat = null;
    }

    $status = 'needs_parsing';
    $isparsed = false;
    if ($sources <= 0) {
        $status = 'no_materials';
    } else if ($signature !== '' && $manifestsignature !== '' && $signature === $manifestsignature) {
        $status = 'parsed';
        $isparsed = true;
    }

    return [
        'status' => $status,
        'is_parsed' => $isparsed,
        'sources' => $sources,
        'parsed_at' => $parsedat,
        'signature' => $signature,
        'manifest_signature' => $manifestsignature,
    ];
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
 * Execute Python runner command and return decoded JSON payload.
 *
 * @param array $args
 * @return array
 */
function local_chatbot_run_runner_command(array $args, array $tracecontext = []): array {
    $python = local_chatbot_get_python_path();
    $runner = local_chatbot_resolve_runner_path();
    $started = microtime(true);
    $requestid = isset($tracecontext['request_id']) ? trim((string)$tracecontext['request_id']) : '';
    $questionnumber = isset($tracecontext['question_number']) ? (int)$tracecontext['question_number'] : 0;
    $attempt = isset($tracecontext['attempt']) ? (int)$tracecontext['attempt'] : 0;
    $pagerange = local_chatbot_normalize_page_range(
        isset($tracecontext['page_start']) ? (int)$tracecontext['page_start'] : 0,
        isset($tracecontext['page_end']) ? (int)$tracecontext['page_end'] : 0
    );

    if (!is_file($python)) {
        local_chatbot_trace_log('php_runner_exec_error', [
            'request_id' => $requestid,
            'question_number' => $questionnumber,
            'attempt' => $attempt,
            'duration_ms' => (int)round((microtime(true) - $started) * 1000),
            'error' => 'Python executable not found: ' . $python,
        ], 'error');
        throw new Exception('Python executable not found: ' . $python);
    }
    if (!is_file($runner)) {
        local_chatbot_trace_log('php_runner_exec_error', [
            'request_id' => $requestid,
            'question_number' => $questionnumber,
            'attempt' => $attempt,
            'duration_ms' => (int)round((microtime(true) - $started) * 1000),
            'error' => 'Runner script not found: ' . $runner,
        ], 'error');
        throw new Exception(
            'Runner script not found. Checked path: ' . $runner .
            '. Please set runner file to app/moodle_rag_runner.py in local_chatbot settings.'
        );
    }

    $parts = [local_chatbot_quote_arg($python), local_chatbot_quote_arg($runner)];
    foreach ($args as $arg) {
        $parts[] = local_chatbot_quote_arg((string)$arg);
    }
    $cmd = implode(' ', $parts) . ' 2>&1';
    local_chatbot_trace_log('php_runner_exec_start', [
        'request_id' => $requestid,
        'question_number' => $questionnumber,
        'attempt' => $attempt,
        'page_start' => $pagerange['page_start'],
        'page_end' => $pagerange['page_end'],
        'python_path' => $python,
        'runner_path' => $runner,
    ]);

    $output = [];
    $code = 0;
    exec($cmd, $output, $code);
    $raw = trim(implode("\n", $output));
    $durationms = (int)round((microtime(true) - $started) * 1000);

    if ($code !== 0) {
        local_chatbot_trace_log('php_runner_exec_error', [
            'request_id' => $requestid,
            'question_number' => $questionnumber,
            'attempt' => $attempt,
            'page_start' => $pagerange['page_start'],
            'page_end' => $pagerange['page_end'],
            'duration_ms' => $durationms,
            'exit_code' => $code,
            'error' => 'RAG process failed: ' . $raw,
        ], 'error');
        throw new Exception('RAG process failed: ' . $raw);
    }

    $jsonline = $raw;
    if (strpos($raw, "\n") !== false) {
        $lines = preg_split('/\r\n|\r|\n/', $raw);
        $jsonline = trim((string)end($lines));
    }

    $payload = json_decode($jsonline, true);
    if (!is_array($payload)) {
        local_chatbot_trace_log('php_runner_exec_error', [
            'request_id' => $requestid,
            'question_number' => $questionnumber,
            'attempt' => $attempt,
            'page_start' => $pagerange['page_start'],
            'page_end' => $pagerange['page_end'],
            'duration_ms' => $durationms,
            'exit_code' => $code,
            'error' => 'Invalid runner response: ' . $raw,
        ], 'error');
        throw new Exception('Invalid runner response: ' . $raw);
    }

    local_chatbot_trace_log('php_runner_exec_success', [
        'request_id' => $requestid,
        'question_number' => $questionnumber,
        'attempt' => $attempt,
        'page_start' => $pagerange['page_start'],
        'page_end' => $pagerange['page_end'],
        'duration_ms' => $durationms,
        'exit_code' => $code,
        'answer_chars' => isset($payload['answer']) ? core_text::strlen((string)$payload['answer']) : 0,
        'sources_count' => isset($payload['sources']) && is_array($payload['sources']) ? count($payload['sources']) : 0,
    ]);

    return $payload;
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
function local_chatbot_run_rag_once(string $question, string $mode = 'auto', array $tracecontext = []): array {
    $datadir = local_chatbot_get_data_path();
    $mode = core_text::strtolower(trim($mode));
    if (!in_array($mode, ['auto', 'general', 'general_raw'], true)) {
        $mode = 'auto';
    }

    $requestid = isset($tracecontext['request_id']) ? trim((string)$tracecontext['request_id']) : '';
    $questionnumber = isset($tracecontext['question_number']) ? (int)$tracecontext['question_number'] : 0;
    $attempt = isset($tracecontext['attempt']) ? (int)$tracecontext['attempt'] : 0;
    $pagerange = local_chatbot_normalize_page_range(
        isset($tracecontext['page_start']) ? (int)$tracecontext['page_start'] : 0,
        isset($tracecontext['page_end']) ? (int)$tracecontext['page_end'] : 0
    );
    $questionb64 = base64_encode($question);
    $runnerargs = [
        '--data-dir',
        $datadir,
        '--query-b64',
        $questionb64,
        '--mode',
        $mode,
        '--trace-log',
        local_chatbot_get_python_trace_log_path(),
    ];
    if ($requestid !== '') {
        $runnerargs[] = '--request-id';
        $runnerargs[] = $requestid;
    }
    if ($questionnumber > 0) {
        $runnerargs[] = '--question-number';
        $runnerargs[] = (string)$questionnumber;
    }
    if ($attempt > 0) {
        $runnerargs[] = '--attempt';
        $runnerargs[] = (string)$attempt;
    }
    if ($pagerange['page_start'] > 0) {
        $runnerargs[] = '--page-start';
        $runnerargs[] = (string)$pagerange['page_start'];
    }
    if ($pagerange['page_end'] > 0) {
        $runnerargs[] = '--page-end';
        $runnerargs[] = (string)$pagerange['page_end'];
    }

    $payload = local_chatbot_run_runner_command($runnerargs, $tracecontext);
    if (!array_key_exists('answer', $payload)) {
        throw new Exception('Invalid runner response: missing answer field.');
    }

    return [
        'answer' => (string)$payload['answer'],
        'sources' => isset($payload['sources']) && is_array($payload['sources']) ? $payload['sources'] : [],
    ];
}

/**
 * Run pre-parse/index warmup for current chatbot data directory.
 *
 * @return array{ok:bool,preparsed:bool,rebuilt:bool,sources:int}
 */
function local_chatbot_preparse_data_dir_documents(): array {
    $datadir = local_chatbot_get_data_path();
    local_chatbot_ensure_data_dir();
    $payload = local_chatbot_run_runner_command([
        '--data-dir',
        $datadir,
        '--preparse',
    ]);

    return [
        'ok' => !empty($payload['ok']),
        'preparsed' => !empty($payload['preparsed']),
        'rebuilt' => !empty($payload['rebuilt']),
        'sources' => (int)($payload['sources'] ?? 0),
    ];
}

/**
 * Sync one course/topic materials then warm vector index cache.
 *
 * @param int $courseid
 * @param int $userid
 * @param string $topic
 * @return array{ok:bool,synced:int,preparsed:bool,rebuilt:bool,sources:int}
 */
function local_chatbot_preparse_course_materials(int $courseid, int $userid, string $topic = ''): array {
    $synced = local_chatbot_sync_course_topic_materials_to_data($courseid, $userid, $topic);
    if (empty($synced)) {
        return [
            'ok' => true,
            'synced' => 0,
            'preparsed' => false,
            'rebuilt' => false,
            'sources' => 0,
        ];
    }

    $prep = local_chatbot_preparse_data_dir_documents();
    return [
        'ok' => !empty($prep['ok']),
        'synced' => count($synced),
        'preparsed' => !empty($prep['preparsed']),
        'rebuilt' => !empty($prep['rebuilt']),
        'sources' => (int)($prep['sources'] ?? 0),
    ];
}

/**
 * Runs Python RAG runner and retries once for long prompts if response is generic fallback.
 *
 * @param string $question
 * @return array
 */
function local_chatbot_run_rag(string $question, array $tracecontext = []): array {
    $result = local_chatbot_run_rag_once($question, 'auto', $tracecontext);
    $normalizedquestion = core_text::strtolower(trim($question));
    $islongprompt = core_text::strlen(trim($question)) >= 80;
    $issimplegreeting = in_array($normalizedquestion, ['hi', 'hello', 'halo', 'hey'], true);

    if (!$issimplegreeting && $islongprompt && local_chatbot_is_generic_fallback_answer((string)$result['answer'])) {
        $result = local_chatbot_run_rag_once($question, 'auto', $tracecontext);
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
function local_chatbot_run_llm_general(string $prompt, bool $rawanswer = false, array $tracecontext = []): array {
    $mode = $rawanswer ? 'general_raw' : 'general';
    return local_chatbot_run_rag_once($prompt, $mode, $tracecontext);
}

/**
 * Build a conversational prompt for follow-up questions using recent chat turns.
 *
 * @param string $question
 * @param array $history
 * @return string
 */
function local_chatbot_build_chat_request_prompt(string $question, array $history = []): string {
    $question = trim($question);
    if ($question === '') {
        return '';
    }

    $lines = [];
    $maxentries = 6;
    $entries = array_slice(array_values($history), -$maxentries);
    foreach ($entries as $entry) {
        if (!is_array($entry)) {
            continue;
        }
        $role = core_text::strtolower(trim((string)($entry['role'] ?? '')));
        $text = trim((string)($entry['text'] ?? ''));
        if ($text === '') {
            continue;
        }
        if ($role !== 'user' && $role !== 'assistant') {
            continue;
        }
        if (core_text::strlen($text) > 600) {
            $text = core_text::substr($text, 0, 600) . '...';
        }
        $label = $role === 'assistant' ? 'Assistant' : 'User';
        $lines[] = $label . ': ' . $text;
    }

    if (empty($lines)) {
        return $question;
    }

    return "Conversation context:\n"
        . implode("\n", $lines)
        . "\n\nCurrent user question:\n"
        . $question
        . "\n\nAnswer the current user question. If it refers to the previous topic, continue that topic.";
}

/**
 * Detects whether the user has a teacher-like role assignment.
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
 * Detects whether user has student role assignment in any context.
 *
 * @param int $userid
 * @return bool
 */
function local_chatbot_user_is_student_like(int $userid): bool {
    global $DB;

    if ($userid <= 0) {
        return false;
    }

    if (is_siteadmin($userid)) {
        return false;
    }

    $sql = "SELECT 1
              FROM {role_assignments} ra
              JOIN {role} r ON r.id = ra.roleid
             WHERE ra.userid = :userid
               AND " . $DB->sql_compare_text('r.shortname') . " = :student
          ORDER BY ra.id ASC";
    $params = [
        'userid' => $userid,
        'student' => 'student',
    ];
    return (bool)$DB->record_exists_sql($sql, $params);
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
            $modified = (int)$file->get_timemodified();
            if ($modified > 0) {
                @touch($targetpath, $modified);
            }
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
