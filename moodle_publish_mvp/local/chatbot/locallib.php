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
 * Read simple KEY=VALUE pairs from the project .env file.
 *
 * @return array
 */
function local_chatbot_read_project_env_file(): array {
    $envpath = local_chatbot_get_project_path() . DIRECTORY_SEPARATOR . '.env';
    if (!is_file($envpath)) {
        return [];
    }

    $lines = @file($envpath, FILE_IGNORE_NEW_LINES);
    if (!is_array($lines)) {
        return [];
    }

    $values = [];
    foreach ($lines as $rawline) {
        $line = trim((string)$rawline);
        if ($line === '' || strpos($line, '#') === 0) {
            continue;
        }
        $pos = strpos($line, '=');
        if ($pos === false) {
            continue;
        }
        $key = trim(substr($line, 0, $pos));
        $value = trim(substr($line, $pos + 1));
        if ($key === '') {
            continue;
        }
        if (strlen($value) >= 2) {
            $first = substr($value, 0, 1);
            $last = substr($value, -1);
            if (($first === '"' || $first === "'") && $first === $last) {
                $value = substr($value, 1, -1);
            }
        }
        $values[$key] = $value;
    }

    return $values;
}

/**
 * Return embedding configuration visible to the Moodle UI.
 *
 * @return array
 */
function local_chatbot_get_embedding_runtime_config(): array {
    $env = local_chatbot_read_project_env_file();

    return [
        'default_backend' => trim((string)($env['EMBED_BACKEND'] ?? 'auto')),
        'ollama_model' => trim((string)($env['EMBED_MODEL'] ?? 'nomic-embed-text')),
        'bert_model' => trim((string)($env['BERT_MODEL'] ?? 'sentence-transformers/msmarco-bert-base-dot-v5')),
    ];
}

/**
 * Return plugin string or fallback when lang key is missing.
 *
 * @param string $key
 * @param string $fallback
 * @return string
 */
function local_chatbot_get_string_or_fallback(string $key, string $fallback): string {
    $value = get_string($key, 'local_chatbot');
    if (preg_match('/^\[\[[^\]]+\]\]$/', $value)) {
        return $fallback;
    }
    return $value;
}

/**
 * Resolve the default chat mode from the current embedding runtime config.
 *
 * @param array|null $embeddingconfig
 * @return string
 */
function local_chatbot_get_default_chat_mode(?array $embeddingconfig = null): string {
    $config = is_array($embeddingconfig) ? $embeddingconfig : local_chatbot_get_embedding_runtime_config();
    $defaultchatmode = 'rag_ollama';
    if (($config['default_backend'] ?? '') === 'bert') {
        $defaultchatmode = stripos((string)($config['bert_model'] ?? ''), 'msmarco') !== false
            ? 'rag_msmarco'
            : 'rag_bert';
    }
    return $defaultchatmode;
}

/**
 * Build the frontend boot payload used by the rewritten chat app shell.
 *
 * @param int $userid
 * @param string $sesskey
 * @param array $coursetopicsmap
 * @param bool $canmanualupload
 * @param bool $canrefreshembedding
 * @return array
 */
function local_chatbot_build_frontend_boot_config(
    int $userid,
    string $sesskey,
    array $coursetopicsmap = [],
    bool $canmanualupload = false,
    bool $canrefreshembedding = false
): array {
    $embeddingconfig = local_chatbot_get_embedding_runtime_config();
    $defaultchatmode = local_chatbot_get_default_chat_mode($embeddingconfig);

    $embeddingconfigtitle = local_chatbot_get_string_or_fallback('embeddingconfigtitle', 'Embedding configuration');
    $embeddingconfigactive = local_chatbot_get_string_or_fallback('embeddingconfigactive', 'Active embedding');
    $embeddingconfigbackend = local_chatbot_get_string_or_fallback('embeddingconfigbackend', 'Default backend');
    $embeddingconfigollama = local_chatbot_get_string_or_fallback('embeddingconfigollama', 'Ollama embedding model');
    $embeddingconfigbert = local_chatbot_get_string_or_fallback('embeddingconfigbert', 'BERT embedding model');
    $embeddingconfigllmonly = local_chatbot_get_string_or_fallback(
        'embeddingconfigllmonly',
        'No embedding is used in LLM-only mode.'
    );
    $refreshembeddingbutton = local_chatbot_get_string_or_fallback('refreshembeddingbutton', 'Refresh embedding');
    $refreshembeddingloading = local_chatbot_get_string_or_fallback(
        'refreshembeddingloading',
        'Refreshing embedding index...'
    );
    $refreshembeddingrequired = local_chatbot_get_string_or_fallback(
        'refreshembeddingrequired',
        'Select a document first.'
    );
    $refreshembeddingok = local_chatbot_get_string_or_fallback(
        'refreshembeddingok',
        'Embedding index refreshed for the active corpus.'
    );
    $refreshembeddingerror = local_chatbot_get_string_or_fallback(
        'refreshembeddingerror',
        'Failed to refresh embedding index.'
    );

    $classplaceholder = local_chatbot_get_string_or_fallback('classplaceholder', 'Select class');
    $topicplaceholder = local_chatbot_get_string_or_fallback('topicplaceholder', 'Select topic');

    $activeembeddingtext = 'Ollama: ' . (string)$embeddingconfig['ollama_model'];
    if ($defaultchatmode === 'llm_only') {
        $activeembeddingtext = $embeddingconfigllmonly;
    } else if ($defaultchatmode === 'rag_bert' || $defaultchatmode === 'rag_msmarco') {
        $activeembeddingtext = 'BERT: ' . (string)$embeddingconfig['bert_model'];
    }

    return [
        'bootversion' => 2,
        'approotid' => 'local-chatbot-app',
        'renderermode' => 'php-shell-app',
        'appownschat' => true,
        'appownsmaterialspreview' => true,
        'ajaxurl' => (new moodle_url('/local/chatbot/ajax.php'))->out(false),
        'sesskey' => $sesskey,
        'userid' => $userid,
        'chaterror' => get_string('chaterror', 'local_chatbot'),
        'nofiles' => get_string('nofiles', 'local_chatbot'),
        'defaultgreeting' => get_string('defaultgreeting', 'local_chatbot'),
        'thinking' => get_string('thinking', 'local_chatbot'),
        'chatusagelabel' => get_string('chatusagelabel', 'local_chatbot'),
        'previewempty' => get_string('previewempty', 'local_chatbot'),
        'previewloading' => get_string('previewloading', 'local_chatbot'),
        'previewerror' => get_string('previewerror', 'local_chatbot'),
        'previewopenpdf' => get_string('previewopenpdf', 'local_chatbot'),
        'previewpdffallback' => get_string('previewpdffallback', 'local_chatbot'),
        'clearhistoryconfirm' => get_string('clearhistoryconfirm', 'local_chatbot'),
        'statusready' => get_string('statusready', 'local_chatbot'),
        'statusnodocs' => get_string('statusnodocs', 'local_chatbot'),
        'modellabel' => get_string('modellabel', 'local_chatbot'),
        'modeplaceholder' => get_string('modeplaceholder', 'local_chatbot'),
        'mode_llm_only' => get_string('mode_llm_only', 'local_chatbot'),
        'mode_rag_ollama' => get_string('mode_rag_ollama', 'local_chatbot'),
        'mode_rag_bert' => get_string('mode_rag_bert', 'local_chatbot'),
        'mode_rag_msmarco' => get_string('mode_rag_msmarco', 'local_chatbot'),
        'embeddingconfigtitle' => $embeddingconfigtitle,
        'embeddingconfigactive' => $embeddingconfigactive,
        'embeddingconfigbackend' => $embeddingconfigbackend,
        'embeddingconfigollama' => $embeddingconfigollama,
        'embeddingconfigbert' => $embeddingconfigbert,
        'embeddingconfigllmonly' => $embeddingconfigllmonly,
        'embedbackenddefault' => (string)$embeddingconfig['default_backend'],
        'embedmodelollama' => (string)$embeddingconfig['ollama_model'],
        'embedmodelbert' => (string)$embeddingconfig['bert_model'],
        'defaultchatmode' => $defaultchatmode,
        'activeembeddingtext' => $activeembeddingtext,
        'refreshembeddingbutton' => $refreshembeddingbutton,
        'refreshembeddingloading' => $refreshembeddingloading,
        'refreshembeddingrequired' => $refreshembeddingrequired,
        'refreshembeddingok' => $refreshembeddingok,
        'refreshembeddingerror' => $refreshembeddingerror,
        'manualuploadrequired' => get_string('manualuploadrequired', 'local_chatbot'),
        'manualuploading' => get_string('manualuploading', 'local_chatbot'),
        'manualuploadsuccess' => get_string('manualuploadsuccess', 'local_chatbot'),
        'manualcleared' => get_string('manualcleared', 'local_chatbot'),
        'manualmodeactive' => get_string('manualmodeactive', 'local_chatbot'),
        'manualuploadreadonly' => get_string('manualuploadreadonly', 'local_chatbot'),
        'canmanualupload' => $canmanualupload,
        'canrefreshembedding' => $canrefreshembedding,
        'courseclassplaceholder' => $classplaceholder,
        'coursetopicplaceholder' => $topicplaceholder,
        'coursetopics' => $coursetopicsmap,
    ];
}

/**
 * Resolve cache namespace name used by the Python runner for an embedding backend/model pair.
 *
 * @param string $backend
 * @param string $modelname
 * @return string
 */
function local_chatbot_resolve_embedding_cache_namespace(string $backend, string $modelname): string {
    $normalizedbackend = core_text::strtolower(trim($backend));
    if ($normalizedbackend === '') {
        return '';
    }

    $normalizedmodel = core_text::strtolower(trim($modelname));
    if ($normalizedmodel === '') {
        return $normalizedbackend;
    }

    $safemodel = preg_replace('/[^a-z0-9._-]+/', '_', $normalizedmodel);
    return $normalizedbackend . '_' . $safemodel;
}

/**
 * Resolve current embedding backend/model details from project config.
 *
 * @return array{configured_backend:string,resolved_backend:string,model_name:string,cache_namespace:string}
 */
function local_chatbot_get_current_embedding_runtime_details(): array {
    $config = local_chatbot_get_embedding_runtime_config();
    $configuredbackend = core_text::strtolower(trim((string)($config['default_backend'] ?? 'auto')));
    if (!in_array($configuredbackend, ['auto', 'bert', 'ollama'], true)) {
        $configuredbackend = 'auto';
    }

    $resolvedbackend = $configuredbackend;
    if ($resolvedbackend === 'auto') {
        $resolvedbackend = trim((string)($config['bert_model'] ?? '')) !== '' ? 'bert' : 'ollama';
    }

    $modelname = $resolvedbackend === 'bert'
        ? trim((string)($config['bert_model'] ?? ''))
        : trim((string)($config['ollama_model'] ?? ''));

    return [
        'configured_backend' => $configuredbackend,
        'resolved_backend' => $resolvedbackend,
        'model_name' => $modelname,
        'cache_namespace' => local_chatbot_resolve_embedding_cache_namespace($resolvedbackend, $modelname),
    ];
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
 * Resolve system-evaluation plotting script path.
 *
 * @return string
 */
function local_chatbot_resolve_system_eval_plot_script_path(): string {
    $projectpath = local_chatbot_get_project_path();
    $candidates = [
        $projectpath . DIRECTORY_SEPARATOR . 'scripts' . DIRECTORY_SEPARATOR . 'eval' . DIRECTORY_SEPARATOR . 'plot_system_eval.py',
        $projectpath . DIRECTORY_SEPARATOR . 'scripts' . DIRECTORY_SEPARATOR . 'plot_system_eval.py',
        $projectpath . DIRECTORY_SEPARATOR . 'plot_system_eval.py',
    ];
    foreach ($candidates as $candidate) {
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
 * Returns LLM answer-runs JSONL path inside project data directory.
 *
 * @return string
 */
function local_chatbot_get_eval_results_path(): string {
    return local_chatbot_get_project_path()
        . DIRECTORY_SEPARATOR . 'data'
        . DIRECTORY_SEPARATOR . 'answer_runs'
        . DIRECTORY_SEPARATOR . 'llm_answer_results.jsonl';
}

/**
 * Returns system-performance evaluation results directory.
 *
 * @return string
 */
function local_chatbot_get_system_eval_results_dir(): string {
    return local_chatbot_get_project_path()
        . DIRECTORY_SEPARATOR . 'data'
        . DIRECTORY_SEPARATOR . 'system_eval_results';
}

/**
 * Create a new unique evaluation session output path.
 *
 * @param string $prefix
 * @return string
 */
function local_chatbot_create_eval_results_session_path(string $prefix = 'answer_runs_dataset'): string {
    $dir = dirname(local_chatbot_get_eval_results_path());
    if (!is_dir($dir)) {
        @mkdir($dir, 0777, true);
    }
    $safe = preg_replace('/[^a-zA-Z0-9_-]+/', '_', trim($prefix));
    if ($safe === null || $safe === '') {
        $safe = 'answer_runs_dataset';
    }
    try {
        $random = bin2hex(random_bytes(3));
    } catch (Throwable $e) {
        $random = substr(sha1(uniqid('', true)), 0, 6);
    }
    $name = $safe . '_' . gmdate('Ymd_His') . '_' . $random . '.jsonl';
    return $dir . DIRECTORY_SEPARATOR . $name;
}

/**
 * Create a unique system-evaluation result path.
 *
 * @param string $prefix
 * @param string $extension
 * @return string
 */
function local_chatbot_create_system_eval_results_path(string $prefix, string $extension): string {
    $dir = local_chatbot_get_system_eval_results_dir();
    if (!is_dir($dir)) {
        @mkdir($dir, 0777, true);
    }
    $safe = preg_replace('/[^a-zA-Z0-9_-]+/', '_', trim($prefix));
    if ($safe === null || $safe === '') {
        $safe = 'objective_eval';
    }
    $ext = preg_replace('/[^a-zA-Z0-9]+/', '', trim($extension));
    if ($ext === null || $ext === '') {
        $ext = 'json';
    }
    try {
        $random = bin2hex(random_bytes(3));
    } catch (Throwable $e) {
        $random = substr(sha1(uniqid('', true)), 0, 6);
    }
    return $dir . DIRECTORY_SEPARATOR . $safe . '_' . gmdate('Ymd_His') . '_' . $random . '.' . $ext;
}

/**
 * Append one evaluation payload line to JSONL file.
 *
 * @param string $path
 * @param array $payload
 * @return void
 */
function local_chatbot_append_eval_payload_jsonl(string $path, array $payload): void {
    if ($path === '') {
        return;
    }
    $dir = dirname($path);
    if (!is_dir($dir)) {
        @mkdir($dir, 0777, true);
    }
    $line = json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_INVALID_UTF8_SUBSTITUTE);
    if ($line === false) {
        return;
    }
    @file_put_contents($path, $line . PHP_EOL, FILE_APPEND | LOCK_EX);
}

/**
 * Write a JSON file payload.
 *
 * @param string $path
 * @param array $payload
 * @return void
 */
function local_chatbot_write_json_file(string $path, array $payload): void {
    if ($path === '') {
        return;
    }
    $dir = dirname($path);
    if (!is_dir($dir)) {
        @mkdir($dir, 0777, true);
    }
    $json = json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_INVALID_UTF8_SUBSTITUTE);
    if ($json === false) {
        return;
    }
    @file_put_contents($path, $json);
}

/**
 * Run Python plotting script for objective system-evaluation summary.
 *
 * @param string $summarypath
 * @return array
 */
function local_chatbot_run_system_eval_plotting(string $summarypath): array {
    $python = local_chatbot_get_python_path();
    $script = local_chatbot_resolve_system_eval_plot_script_path();
    if ($summarypath === '' || !is_file($summarypath)) {
        throw new moodle_exception('Objective evaluation summary file not found for plotting.');
    }
    if (!is_file($python)) {
        throw new moodle_exception('Configured Python executable for plotting was not found.');
    }
    if (!is_file($script)) {
        throw new moodle_exception('System-evaluation plotting script was not found.');
    }

    $command = escapeshellarg($python)
        . ' '
        . escapeshellarg($script)
        . ' --summary '
        . escapeshellarg($summarypath);

    $output = [];
    $exitcode = 0;
    @exec($command . ' 2>&1', $output, $exitcode);
    if ($exitcode !== 0) {
        throw new moodle_exception('System-evaluation plotting failed: ' . implode("\n", $output));
    }

    $lastline = '';
    if (!empty($output)) {
        $lastline = trim((string)end($output));
    }
    $decoded = json_decode($lastline, true);
    if (!is_array($decoded)) {
        throw new moodle_exception('System-evaluation plotting returned invalid JSON.');
    }
    return $decoded;
}

/**
 * Load answer-run JSONL rows from file.
 *
 * @param string $path
 * @return array
 */
function local_chatbot_load_answer_runs_jsonl(string $path): array {
    if ($path === '' || !is_file($path)) {
        return [];
    }
    $lines = @file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    if (!is_array($lines)) {
        return [];
    }
    $rows = [];
    foreach ($lines as $line) {
        $decoded = json_decode((string)$line, true);
        if (is_array($decoded)) {
            $rows[] = $decoded;
        }
    }
    return $rows;
}

/**
 * Normalize question scope.
 *
 * @param string $value
 * @return string
 */
function local_chatbot_normalize_eval_scope(string $value): string {
    $text = core_text::strtolower(trim($value));
    if (in_array($text, ['in-scope', 'inscope', 'in_scope', 'answerable'], true)) {
        return 'in-scope';
    }
    if (in_array($text, ['out-of-scope', 'outofscope', 'out_of_scope', 'unanswerable'], true)) {
        return 'out-of-scope';
    }
    return $text !== '' ? $text : 'unknown';
}

/**
 * Normalize expected system behavior.
 *
 * @param string $value
 * @param string $scope
 * @return string
 */
function local_chatbot_normalize_expected_behavior(string $value, string $scope): string {
    $text = core_text::strtolower(trim($value));
    if (in_array($text, ['answer', 'refuse'], true)) {
        return $text;
    }
    if ($scope === 'in-scope') {
        return 'answer';
    }
    if ($scope === 'out-of-scope') {
        return 'refuse';
    }
    return 'unknown';
}

/**
 * Normalize source filename for matching.
 *
 * @param string $value
 * @return string
 */
function local_chatbot_normalize_source_name(string $value): string {
    $trimmed = trim($value);
    if ($trimmed === '') {
        return '';
    }
    return core_text::strtolower(basename(str_replace(['/', '\\'], DIRECTORY_SEPARATOR, $trimmed)));
}

/**
 * Extract PDF page range from legacy gold_source text.
 *
 * @param string $text
 * @return array
 */
function local_chatbot_extract_gold_source_page_range(string $text): array {
    $matches = [];
    if (!preg_match('/PDF\s+page\s+(\d+)(?:\s*[-to]+\s*(\d+))?/i', $text, $matches)) {
        return [null, null];
    }
    $start = (int)$matches[1];
    $end = isset($matches[2]) && $matches[2] !== '' ? (int)$matches[2] : $start;
    return [$start, $end];
}

/**
 * Normalize gold sources for objective system evaluation.
 *
 * @param array $item
 * @param string $defaultsource
 * @return array
 */
function local_chatbot_normalize_gold_sources(array $item, string $defaultsource): array {
    $systemeval = [];
    if (isset($item['system_eval']) && is_array($item['system_eval'])) {
        $systemeval = $item['system_eval'];
    }
    $rawstructured = [];
    if (isset($systemeval['gold_sources']) && is_array($systemeval['gold_sources'])) {
        $rawstructured = $systemeval['gold_sources'];
    } else if (isset($item['gold_sources']) && is_array($item['gold_sources'])) {
        $rawstructured = $item['gold_sources'];
    }

    $normalized = [];
    if (!empty($rawstructured)) {
        foreach ($rawstructured as $entry) {
            if (!is_array($entry)) {
                continue;
            }
            $source = local_chatbot_normalize_source_name((string)($entry['source'] ?? $defaultsource));
            if ($source === '') {
                continue;
            }
            $pagestart = isset($entry['page_start']) && $entry['page_start'] !== '' ? (int)$entry['page_start'] : null;
            $pageend = isset($entry['page_end']) && $entry['page_end'] !== '' ? (int)$entry['page_end'] : $pagestart;
            $normalized[] = [
                'source' => $source,
                'page_start' => $pagestart,
                'page_end' => $pageend,
            ];
        }
        return $normalized;
    }

    $rawlegacy = $item['gold_source'] ?? [];
    if (is_string($rawlegacy)) {
        $rawlegacy = [$rawlegacy];
    }
    if (!is_array($rawlegacy)) {
        return [];
    }

    foreach ($rawlegacy as $entry) {
        list($pagestart, $pageend) = local_chatbot_extract_gold_source_page_range((string)$entry);
        $source = local_chatbot_normalize_source_name($defaultsource);
        if ($source === '' && $pagestart === null && $pageend === null) {
            continue;
        }
        $normalized[] = [
            'source' => $source,
            'page_start' => $pagestart,
            'page_end' => $pageend,
        ];
    }
    return $normalized;
}

/**
 * Build normalized question specs for objective system evaluation.
 *
 * @param array $questions
 * @return array
 */
function local_chatbot_build_objective_question_specs(array $questions): array {
    $specs = [];
    $topscope = '';
    $defaultsource = '';
    if (isset($questions['scope'])) {
        $topscope = (string)$questions['scope'];
    }
    if (isset($questions['source_document'])) {
        $defaultsource = local_chatbot_normalize_source_name((string)$questions['source_document']);
    }

    foreach ($questions as $index => $item) {
        if (!is_array($item)) {
            continue;
        }
        $question = trim((string)($item['question'] ?? ''));
        if ($question === '') {
            continue;
        }
        $questionid = trim((string)($item['question_id'] ?? $item['id'] ?? ('auto-q' . str_pad((string)((int)$index + 1), 3, '0', STR_PAD_LEFT))));
        $scope = local_chatbot_normalize_eval_scope((string)($item['scope'] ?? $topscope));
        $systemeval = isset($item['system_eval']) && is_array($item['system_eval']) ? $item['system_eval'] : [];
        $expectedbehavior = local_chatbot_normalize_expected_behavior((string)($systemeval['expected_behavior'] ?? ($item['expected_behavior'] ?? '')), $scope);
        $specs[$questionid] = [
            'question_id' => $questionid,
            'question' => $question,
            'scope' => $scope,
            'expected_behavior' => $expectedbehavior,
            'gold_sources' => local_chatbot_normalize_gold_sources($item, $defaultsource),
        ];
    }
    return $specs;
}

/**
 * Infer predicted system behavior from final answer text.
 *
 * @param string $answer
 * @param string $status
 * @return string
 */
function local_chatbot_infer_predicted_behavior(string $answer, string $status): string {
    if (core_text::strtolower(trim($status)) !== 'success') {
        return 'error';
    }
    $text = core_text::strtolower(trim($answer));
    if ($text === '') {
        return 'error';
    }
    $patterns = [
        'not found in context',
        'not found in the provided context',
        'not found in the provided material',
        'the context does not contain',
        'the provided context does not contain',
        'the material does not contain',
        'the provided material does not contain',
        'cannot answer from the provided material',
        'cannot answer from the material',
        'cannot be answered from the provided material',
        'cannot be answered from the material',
        'insufficient context',
        'insufficient information in the provided context',
        'no relevant information in the provided context',
    ];
    foreach ($patterns as $pattern) {
        if (strpos($text, $pattern) !== false) {
            return 'refuse';
        }
    }
    return 'answer';
}

/**
 * Build an automatic online system-performance snapshot for one chat response.
 *
 * This snapshot intentionally uses only metrics that are available at runtime
 * for arbitrary user queries, without assuming gold answers or gold sources.
 *
 * @param array $payload
 * @param array $context
 * @return array
 */
function local_chatbot_build_online_eval_snapshot(array $payload, array $context = []): array {
    $answer = isset($payload['answer']) ? (string)$payload['answer'] : '';
    $sources = [];
    if (isset($payload['sources']) && is_array($payload['sources'])) {
        $sources = array_values(array_map(static function($item): string {
            return trim((string)$item);
        }, $payload['sources']));
    }

    $status = core_text::strtolower(trim((string)($payload['status'] ?? ($context['status'] ?? 'success'))));
    if ($status === '') {
        $status = 'success';
    }

    $chatmode = trim((string)($context['chat_mode'] ?? ($payload['mode'] ?? '')));
    $retrievedcontextcount = max(0, (int)($payload['retrieved_context_count'] ?? 0));
    $snapshot = [
        'request_id' => trim((string)($context['request_id'] ?? '')),
        'userid' => max(0, (int)($context['userid'] ?? 0)),
        'courseid' => max(0, (int)($context['courseid'] ?? 0)),
        'chat_mode' => $chatmode,
        'question_id' => trim((string)($context['question_id'] ?? ($payload['question_id'] ?? ''))),
        'run_id' => max(0, (int)($context['run_id'] ?? ($payload['run_id'] ?? 0))),
        'topic' => trim((string)($context['topic'] ?? '')),
        'question_text' => trim((string)($context['question_text'] ?? '')),
        'answer_text' => $answer,
        'sources' => $sources,
        'status' => $status,
        'predicted_behavior' => local_chatbot_infer_predicted_behavior($answer, $status),
        'model_name' => trim((string)($payload['model_name'] ?? '')),
        'embedding_backend' => trim((string)($payload['embedding_backend'] ?? '')),
        'embedding_model_name' => trim((string)($payload['embedding_model_name'] ?? '')),
        'latency_total' => max(0.0, (float)($payload['latency_total'] ?? 0.0)),
        'latency_retrieval' => max(0.0, (float)($payload['latency_retrieval'] ?? 0.0)),
        'latency_generation' => max(0.0, (float)($payload['latency_generation'] ?? 0.0)),
        'retrieved_context_count' => $retrievedcontextcount,
        'source_count' => count(array_filter($sources, static function($item): bool {
            return $item !== '';
        })),
        'answer_chars' => core_text::strlen($answer),
        'error_message' => trim((string)($payload['error_message'] ?? ($context['error_message'] ?? ''))),
    ];

    return $snapshot;
}

/**
 * Build an error snapshot when a chat request fails before a normal response.
 *
 * @param array $context
 * @return array
 */
function local_chatbot_build_online_eval_error_snapshot(array $context = []): array {
    return local_chatbot_build_online_eval_snapshot([
        'answer' => '',
        'sources' => [],
        'status' => 'error',
        'error_message' => trim((string)($context['error_message'] ?? '')),
        'latency_total' => max(0.0, (float)($context['latency_total'] ?? 0.0)),
        'latency_retrieval' => max(0.0, (float)($context['latency_retrieval'] ?? 0.0)),
        'latency_generation' => max(0.0, (float)($context['latency_generation'] ?? 0.0)),
        'retrieved_context_count' => max(0, (int)($context['retrieved_context_count'] ?? 0)),
    ], $context);
}

/**
 * Check whether a retrieved context item matches a gold source target.
 *
 * @param array $goldsource
 * @param array $contextitem
 * @return bool
 */
function local_chatbot_gold_source_matches_context(array $goldsource, array $contextitem): bool {
    $goldname = local_chatbot_normalize_source_name((string)($goldsource['source'] ?? ''));
    $contextname = local_chatbot_normalize_source_name((string)($contextitem['source'] ?? ''));
    if ($goldname !== '' && $contextname !== '' && $goldname !== $contextname) {
        return false;
    }

    $page = isset($contextitem['page']) && $contextitem['page'] !== '' ? (int)$contextitem['page'] : null;
    $pagestart = isset($goldsource['page_start']) && $goldsource['page_start'] !== '' ? (int)$goldsource['page_start'] : null;
    $pageend = isset($goldsource['page_end']) && $goldsource['page_end'] !== '' ? (int)$goldsource['page_end'] : $pagestart;
    if ($pagestart === null && $pageend === null) {
        return true;
    }
    if ($page === null) {
        return false;
    }
    if ($pageend === null) {
        $pageend = $pagestart;
    }
    if ($pagestart === null) {
        $pagestart = $pageend;
    }
    return $pagestart !== null && $pageend !== null && $page >= $pagestart && $page <= $pageend;
}

/**
 * Evaluate objective system-performance metrics for answer runs.
 *
 * @param array $questions
 * @param array $answerruns
 * @param int $topk
 * @return array
 */
function local_chatbot_evaluate_system_performance_rows(array $questions, array $answerruns, int $topk = 4): array {
    $questionspecs = local_chatbot_build_objective_question_specs($questions);
    $rows = [];
    $effectivetopk = max(1, $topk);

    foreach ($answerruns as $run) {
        if (!is_array($run)) {
            continue;
        }
        $questionid = trim((string)($run['question_id'] ?? ''));
        if ($questionid === '' || !isset($questionspecs[$questionid])) {
            continue;
        }
        $spec = $questionspecs[$questionid];
        $status = core_text::strtolower(trim((string)($run['status'] ?? 'success')));
        $answer = (string)($run['model_answer'] ?? '');
        $predictedbehavior = local_chatbot_infer_predicted_behavior($answer, $status);
        $retrievedcontext = isset($run['retrieved_context']) && is_array($run['retrieved_context']) ? $run['retrieved_context'] : [];
        $topcontext = array_slice($retrievedcontext, 0, $effectivetopk);
        $goldsources = isset($spec['gold_sources']) && is_array($spec['gold_sources']) ? $spec['gold_sources'] : [];
        $matchedgoldsources = 0;
        $firstmatchrank = null;

        if (!empty($goldsources)) {
            foreach ($goldsources as $goldsource) {
                $foundforsource = false;
                foreach ($topcontext as $index => $contextitem) {
                    if (!is_array($contextitem)) {
                        continue;
                    }
                    if (!local_chatbot_gold_source_matches_context($goldsource, $contextitem)) {
                        continue;
                    }
                    $foundforsource = true;
                    $rank = $index + 1;
                    if ($firstmatchrank === null || $rank < $firstmatchrank) {
                        $firstmatchrank = $rank;
                    }
                    break;
                }
                if ($foundforsource) {
                    $matchedgoldsources++;
                }
            }
        }

        $sourcehitatk = null;
        $sourcerecallatk = null;
        $rankofgoldsource = null;
        $mrr = null;
        if (!empty($goldsources)) {
            $sourcehitatk = $matchedgoldsources > 0 ? 1 : 0;
            $sourcerecallatk = round($matchedgoldsources / count($goldsources), 4);
            $rankofgoldsource = $firstmatchrank;
            $mrr = $firstmatchrank ? round(1.0 / $firstmatchrank, 4) : 0.0;
        }

        $expectedbehavior = (string)($spec['expected_behavior'] ?? 'unknown');
        $answerabledetectioncorrect = null;
        $refusalcorrect = null;
        if (in_array($expectedbehavior, ['answer', 'refuse'], true)) {
            $answerabledetectioncorrect = $predictedbehavior === $expectedbehavior ? 1 : 0;
            if ($expectedbehavior === 'refuse') {
                $refusalcorrect = $answerabledetectioncorrect;
            }
        }

        $rows[] = [
            'question_id' => $questionid,
            'question' => (string)($spec['question'] ?? ($run['question'] ?? '')),
            'mode' => trim((string)($run['mode'] ?? '')),
            'run_id' => (int)($run['run_id'] ?? 0),
            'scope' => (string)($spec['scope'] ?? 'unknown'),
            'expected_behavior' => $expectedbehavior,
            'status' => $status,
            'success_score' => $status === 'success' ? 1 : 0,
            'latency_total' => (float)($run['latency_total'] ?? 0),
            'latency_retrieval' => (float)($run['latency_retrieval'] ?? 0),
            'latency_generation' => (float)($run['latency_generation'] ?? 0),
            'top_k' => $effectivetopk,
            'retrieved_context_count' => count($retrievedcontext),
            'gold_source_count' => count($goldsources),
            'matched_gold_sources' => $matchedgoldsources,
            'source_hit_at_k' => $sourcehitatk,
            'source_recall_at_k' => $sourcerecallatk,
            'rank_of_gold_source' => $rankofgoldsource,
            'mrr' => $mrr,
            'predicted_behavior' => $predictedbehavior,
            'answerable_detection_correct' => $answerabledetectioncorrect,
            'refusal_correct' => $refusalcorrect,
            'timestamp' => (string)($run['timestamp'] ?? gmdate('c')),
        ];
    }

    return $rows;
}

/**
 * Compute average for numeric values.
 *
 * @param array $values
 * @return float|null
 */
function local_chatbot_system_eval_average(array $values): ?float {
    if (empty($values)) {
        return null;
    }
    return round(array_sum($values) / count($values), 4);
}

/**
 * Summarize system-evaluation rows for a mode and optional scope.
 *
 * @param array $rows
 * @param string $mode
 * @param string|null $scope
 * @return array
 */
function local_chatbot_summarize_system_eval_rows(array $rows, string $mode, ?string $scope = null): array {
    $successscores = [];
    $successfulrows = [];
    $sourcehitvalues = [];
    $sourcerecallvalues = [];
    $rankvalues = [];
    $mrrvalues = [];
    $detectionvalues = [];
    $refusalvalues = [];

    foreach ($rows as $row) {
        $successscores[] = (float)($row['success_score'] ?? 0);
        if ((int)($row['success_score'] ?? 0) === 1) {
            $successfulrows[] = $row;
        }
        if (array_key_exists('source_hit_at_k', $row) && $row['source_hit_at_k'] !== null) {
            $sourcehitvalues[] = (float)$row['source_hit_at_k'];
        }
        if (array_key_exists('source_recall_at_k', $row) && $row['source_recall_at_k'] !== null) {
            $sourcerecallvalues[] = (float)$row['source_recall_at_k'];
        }
        if (array_key_exists('rank_of_gold_source', $row) && $row['rank_of_gold_source'] !== null) {
            $rankvalues[] = (float)$row['rank_of_gold_source'];
        }
        if (array_key_exists('mrr', $row) && $row['mrr'] !== null) {
            $mrrvalues[] = (float)$row['mrr'];
        }
        if (array_key_exists('answerable_detection_correct', $row) && $row['answerable_detection_correct'] !== null) {
            $detectionvalues[] = (float)$row['answerable_detection_correct'];
        }
        if (array_key_exists('refusal_correct', $row) && $row['refusal_correct'] !== null) {
            $refusalvalues[] = (float)$row['refusal_correct'];
        }
    }

    $totalsuccesses = 0;
    foreach ($successscores as $score) {
        $totalsuccesses += (int)$score;
    }

    return [
        'mode' => $mode,
        'scope' => $scope,
        'total_runs' => count($rows),
        'successful_runs' => $totalsuccesses,
        'failed_runs' => count($rows) - $totalsuccesses,
        'success_rate' => local_chatbot_system_eval_average($successscores),
        'avg_latency_total' => local_chatbot_system_eval_average(array_map(function($row) {
            return (float)($row['latency_total'] ?? 0);
        }, $successfulrows)),
        'avg_latency_retrieval' => local_chatbot_system_eval_average(array_map(function($row) {
            return (float)($row['latency_retrieval'] ?? 0);
        }, $successfulrows)),
        'avg_latency_generation' => local_chatbot_system_eval_average(array_map(function($row) {
            return (float)($row['latency_generation'] ?? 0);
        }, $successfulrows)),
        'source_hit_at_k_rate' => local_chatbot_system_eval_average($sourcehitvalues),
        'avg_source_recall_at_k' => local_chatbot_system_eval_average($sourcerecallvalues),
        'avg_rank_of_gold_source' => local_chatbot_system_eval_average($rankvalues),
        'mrr' => local_chatbot_system_eval_average($mrrvalues),
        'answerable_detection_accuracy' => local_chatbot_system_eval_average($detectionvalues),
        'refusal_accuracy' => local_chatbot_system_eval_average($refusalvalues),
    ];
}

/**
 * Build aggregated objective system-performance summary.
 *
 * @param array $rows
 * @param int $topk
 * @param string $answerrunsfile
 * @param string $questiondatasetfile
 * @return array
 */
function local_chatbot_build_system_eval_summary(array $rows, int $topk, string $answerrunsfile = '', string $questiondatasetfile = ''): array {
    $modes = [];
    foreach ($rows as $row) {
        $mode = trim((string)($row['mode'] ?? ''));
        if ($mode === '') {
            continue;
        }
        $modes[$mode] = true;
    }
    ksort($modes);

    $bymode = [];
    foreach (array_keys($modes) as $mode) {
        $moderows = array_values(array_filter($rows, function($row) use ($mode) {
            return trim((string)($row['mode'] ?? '')) === $mode;
        }));
        $modesummary = local_chatbot_summarize_system_eval_rows($moderows, $mode, null);
        $scopes = [];
        foreach ($moderows as $row) {
            $scope = (string)($row['scope'] ?? 'unknown');
            $scopes[$scope] = true;
        }
        ksort($scopes);
        $bymodescope = [];
        foreach (array_keys($scopes) as $scope) {
            $scoperows = array_values(array_filter($moderows, function($row) use ($scope) {
                return (string)($row['scope'] ?? 'unknown') === $scope;
            }));
            $bymodescope[] = local_chatbot_summarize_system_eval_rows($scoperows, $mode, $scope);
        }
        $modesummary['by_scope'] = $bymodescope;
        $bymode[] = $modesummary;
    }

    return [
        'generated_at' => gmdate('c'),
        'top_k' => max(1, $topk),
        'answer_runs_file' => $answerrunsfile,
        'question_dataset_file' => $questiondatasetfile,
        'total_runs' => count($rows),
        'by_mode' => $bymode,
    ];
}

/**
 * Run objective system-performance evaluation for a completed answer-run dataset.
 *
 * @param array $questions
 * @param string $answerrunspath
 * @param string $questiondatasetfile
 * @param int $topk
 * @return array
 */
function local_chatbot_run_system_performance_evaluation(array $questions, string $answerrunspath, string $questiondatasetfile = '', int $topk = 4): array {
    $answerruns = local_chatbot_load_answer_runs_jsonl($answerrunspath);
    $rows = local_chatbot_evaluate_system_performance_rows($questions, $answerruns, $topk);
    $perrunpath = local_chatbot_create_system_eval_results_path('objective_eval_runs', 'jsonl');
    foreach ($rows as $row) {
        local_chatbot_append_eval_payload_jsonl($perrunpath, $row);
    }
    $summary = local_chatbot_build_system_eval_summary($rows, $topk, $answerrunspath, $questiondatasetfile);
    $summarypath = local_chatbot_create_system_eval_results_path('objective_eval_summary', 'json');
    local_chatbot_write_json_file($summarypath, $summary);
    try {
        local_chatbot_run_system_eval_plotting($summarypath);
    } catch (Throwable $e) {
        local_chatbot_trace_log('objective_eval_plot_error', [
            'summary_path' => $summarypath,
            'error' => $e->getMessage(),
        ], 'error');
    }

    return [
        'per_run_output_path' => $perrunpath,
        'per_run_output_file' => basename($perrunpath),
        'summary_output_path' => $summarypath,
        'summary_output_file' => basename($summarypath),
        'summary' => $summary,
    ];
}

/**
 * Load evaluation questions from JSON text.
 *
 * @param string $rawjson
 * @return array
 */
function local_chatbot_load_eval_questions_from_json_text(string $rawjson): array {
    $decoded = json_decode($rawjson, true);
    if (!is_array($decoded)) {
        throw new invalid_parameter_exception('Evaluation dataset must be valid JSON.');
    }

    $items = $decoded;
    $topscope = '';
    $topsourcedocument = '';
    if (array_key_exists('questions', $decoded) && is_array($decoded['questions'])) {
        $items = $decoded['questions'];
        $topscope = isset($decoded['scope']) ? (string)$decoded['scope'] : '';
        $topsourcedocument = isset($decoded['source_document']) ? (string)$decoded['source_document'] : '';
    }

    $questions = [];
    $counter = 0;
    foreach ($items as $item) {
        if (!is_array($item)) {
            continue;
        }
        $question = trim((string)($item['question'] ?? ''));
        if ($question === '') {
            continue;
        }
        $counter++;
        $questionid = trim((string)($item['id'] ?? $item['question_id'] ?? ''));
        if ($questionid === '') {
            $questionid = 'auto-q' . str_pad((string)$counter, 3, '0', STR_PAD_LEFT);
        }
        $normalized = $item;
        $normalized['question'] = $question;
        $normalized['question_id'] = $questionid;
        if (!isset($normalized['scope']) && $topscope !== '') {
            $normalized['scope'] = $topscope;
        }
        if (!isset($normalized['source_document']) && $topsourcedocument !== '') {
            $normalized['source_document'] = $topsourcedocument;
        }
        $questions[] = $normalized;
    }

    if (empty($questions)) {
        throw new invalid_parameter_exception('No valid questions found in evaluation dataset.');
    }

    return $questions;
}

/**
 * Normalize one or more chat modes from raw comma-separated input.
 *
 * @param string $rawmodes
 * @return array
 */
function local_chatbot_normalize_chat_modes(string $rawmodes): array {
    $valid = ['llm_only', 'rag_ollama', 'rag_bert', 'rag_msmarco'];
    $parts = preg_split('/[\s,]+/', core_text::strtolower(trim($rawmodes)));
    if (!is_array($parts)) {
        $parts = [];
    }
    $modes = [];
    foreach ($parts as $part) {
        $mode = trim((string)$part);
        if ($mode === '' || !in_array($mode, $valid, true) || isset($modes[$mode])) {
            continue;
        }
        $modes[$mode] = $mode;
    }
    if (empty($modes)) {
        $modes['rag_ollama'] = 'rag_ollama';
    }
    return array_values($modes);
}

/**
 * Run a dataset evaluation session and write outputs to a new JSONL file.
 *
 * @param array $questions
 * @param array $chatmodes
 * @param int $runs
 * @param array $tracecontext
 * @return array
 */
function local_chatbot_run_eval_dataset(
    array $questions,
    array $chatmodes,
    int $runs = 1,
    array $tracecontext = [],
    string $questiondatasetfile = ''
): array {
    $chatmodes = array_values(array_filter($chatmodes, function($mode) {
        return in_array($mode, ['llm_only', 'rag_ollama', 'rag_bert', 'rag_msmarco'], true);
    }));
    if (empty($chatmodes)) {
        $chatmodes = ['rag_ollama'];
    }
    $runs = max(1, $runs);
    $outputpath = local_chatbot_create_eval_results_session_path('answer_runs_dataset');
    $successes = 0;
    $failures = 0;
    $totalruns = 0;

    foreach ($chatmodes as $chatmode) {
        foreach ($questions as $index => $item) {
            if (!is_array($item)) {
                continue;
            }
            $question = trim((string)($item['question'] ?? ''));
            $questionid = trim((string)($item['question_id'] ?? $item['id'] ?? ''));
            if ($question === '' || $questionid === '') {
                continue;
            }

            for ($runid = 1; $runid <= $runs; $runid++) {
                $totalruns++;
                $runtrace = $tracecontext;
                $runtrace['eval_mode'] = true;
                $runtrace['eval_mode_name'] = $chatmode;
                $runtrace['question_id'] = $questionid;
                $runtrace['run_id'] = $runid;
                $runtrace['raw_results_path'] = $outputpath;
                $runtrace['question_number'] = $index + 1;
                $runtrace['attempt'] = $runid;

                if ($chatmode === 'rag_ollama') {
                    $runtrace['embed_backend'] = 'ollama';
                } else if ($chatmode === 'rag_bert' || $chatmode === 'rag_msmarco') {
                    $runtrace['embed_backend'] = 'bert';
                } else {
                    $runtrace['embed_backend'] = 'none';
                }

                try {
                    if ($chatmode === 'llm_only') {
                        local_chatbot_run_llm_general($question, false, $runtrace);
                    } else {
                        local_chatbot_run_rag($question, $runtrace);
                    }
                    $successes++;
                } catch (Throwable $e) {
                    $failures++;
                    local_chatbot_append_eval_payload_jsonl($outputpath, [
                        'question_id' => $questionid,
                        'question' => $question,
                        'mode' => $chatmode,
                        'run_id' => $runid,
                        'model_name' => '',
                        'embedding_backend' => $chatmode === 'llm_only'
                            ? 'none'
                            : (($chatmode === 'rag_bert' || $chatmode === 'rag_msmarco') ? 'bert' : 'ollama'),
                        'model_answer' => '',
                        'retrieved_context' => [],
                        'latency_total' => 0,
                        'latency_retrieval' => 0,
                        'latency_generation' => 0,
                        'status' => 'error',
                        'error_message' => $e->getMessage(),
                        'timestamp' => gmdate('c'),
                    ]);
                    local_chatbot_trace_log('eval_dataset_run_error', [
                        'request_id' => isset($runtrace['request_id']) ? $runtrace['request_id'] : '',
                        'question_id' => $questionid,
                        'run_id' => $runid,
                        'chat_mode' => $chatmode,
                        'error' => $e->getMessage(),
                    ], 'error');
                }
            }
        }
    }

    $objectiveevaluation = [];
    try {
        $objectiveevaluation = local_chatbot_run_system_performance_evaluation($questions, $outputpath, $questiondatasetfile, 4);
    } catch (Throwable $e) {
        $objectiveevaluation = [
            'error' => $e->getMessage(),
        ];
        local_chatbot_trace_log('objective_eval_error', [
            'output_path' => $outputpath,
            'chat_modes' => implode(',', $chatmodes),
            'error' => $e->getMessage(),
        ], 'error');
    }

    return [
        'output_path' => $outputpath,
        'output_file' => basename($outputpath),
        'questions' => count($questions),
        'runs_per_question' => $runs,
        'total_runs' => $totalruns,
        'successes' => $successes,
        'failures' => $failures,
        'chat_mode' => $chatmodes[0],
        'chat_modes' => $chatmodes,
        'objective_evaluation' => $objectiveevaluation,
    ];
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
 * Returns material-context state file path.
 *
 * @return string
 */
function local_chatbot_get_material_context_state_path(): string {
    return local_chatbot_get_data_path() . DIRECTORY_SEPARATOR . '.rag_material_context.json';
}

/**
 * Read persisted material-context state.
 *
 * @return array
 */
function local_chatbot_read_material_context_state(): array {
    $path = local_chatbot_get_material_context_state_path();
    if (!is_file($path)) {
        return [];
    }
    $raw = @file_get_contents($path);
    if ($raw === false || trim($raw) === '') {
        return [];
    }
    $decoded = json_decode($raw, true);
    return is_array($decoded) ? $decoded : [];
}

/**
 * Persist material-context state.
 *
 * @param string $mode
 * @param array $context
 * @return void
 */
function local_chatbot_write_material_context_state(string $mode, array $context = []): void {
    local_chatbot_ensure_data_dir();
    $normalized = core_text::strtolower(trim($mode));
    if (!in_array($normalized, ['none', 'manual', 'topic'], true)) {
        $normalized = 'none';
    }

    $payload = [
        'mode' => $normalized,
        'updated_at' => time(),
    ];
    foreach ($context as $key => $value) {
        $payload[(string)$key] = $value;
    }

    $json = json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT | JSON_INVALID_UTF8_SUBSTITUTE);
    if ($json === false) {
        return;
    }
    @file_put_contents(local_chatbot_get_material_context_state_path(), $json);
}

/**
 * Returns active material-context summary for UI/backend branching.
 *
 * @return array
 */
function local_chatbot_get_material_context_summary(): array {
    $state = local_chatbot_read_material_context_state();
    $files = local_chatbot_list_uploaded_files();
    $sources = count($files);
    $mode = core_text::strtolower(trim((string)($state['mode'] ?? '')));

    if ($sources <= 0) {
        $mode = 'none';
    } else if (!in_array($mode, ['manual', 'topic'], true)) {
        $mode = 'legacy';
    }

    return [
        'mode' => $mode,
        'has_files' => $sources > 0,
        'files_count' => $sources,
        'is_manual' => ($mode === 'manual' && $sources > 0),
        'is_topic' => ($mode === 'topic' && $sources > 0),
        'disable_topic_select' => ($mode === 'manual' && $sources > 0),
        'course_id' => isset($state['course_id']) ? (int)$state['course_id'] : 0,
        'course_name' => trim((string)($state['course_name'] ?? '')),
        'topic' => trim((string)($state['topic'] ?? '')),
        'updated_at' => isset($state['updated_at']) ? (int)$state['updated_at'] : 0,
    ];
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
    return local_chatbot_read_parse_manifest_file($manifestpath);
}

/**
 * Read a parse manifest file from a given path.
 *
 * @param string $manifestpath
 * @return array
 */
function local_chatbot_read_parse_manifest_file(string $manifestpath): array {
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
    $embeddingdetails = local_chatbot_get_current_embedding_runtime_details();
    $datadir = local_chatbot_get_data_path();
    $manifestcandidates = [];
    if ($embeddingdetails['cache_namespace'] !== '') {
        $safe = preg_replace('/[^a-z0-9._-]+/', '_', $embeddingdetails['cache_namespace']);
        $manifestcandidates[] = $datadir . DIRECTORY_SEPARATOR . '.rag_index_manifest_' . $safe . '.json';
    }
    $manifestcandidates[] = $datadir . DIRECTORY_SEPARATOR . '.rag_index_manifest.json';

    $manifest = [];
    foreach ($manifestcandidates as $candidate) {
        $manifest = local_chatbot_read_parse_manifest_file($candidate);
        if (!empty($manifest)) {
            break;
        }
    }

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
        'embedding_backend' => $embeddingdetails['resolved_backend'],
        'embedding_model' => $embeddingdetails['model_name'],
        'configured_backend' => $embeddingdetails['configured_backend'],
    ];
}

/**
 * Get embedding/index status for one selected file within the active corpus.
 *
 * @param string $filename
 * @return array
 */
function local_chatbot_get_file_embedding_status(string $filename): array {
    $cleanfilename = clean_param($filename, PARAM_FILE);
    $parsestatus = local_chatbot_get_current_material_parse_status();
    $activefiles = local_chatbot_list_uploaded_files();
    $isinactivecorpus = false;

    foreach ($activefiles as $file) {
        if ((string)($file['name'] ?? '') === $cleanfilename) {
            $isinactivecorpus = true;
            break;
        }
    }

    return [
        'filename' => $cleanfilename,
        'scope' => 'corpus',
        'file_in_active_corpus' => $isinactivecorpus,
        'is_index_current' => !empty($parsestatus['is_parsed']),
        'is_embedded_in_active_index' => $isinactivecorpus && !empty($parsestatus['is_parsed']),
        'parse_status' => (string)($parsestatus['status'] ?? 'needs_parsing'),
        'parsed_at' => isset($parsestatus['parsed_at']) ? (int)$parsestatus['parsed_at'] : 0,
        'sources' => (int)($parsestatus['sources'] ?? 0),
        'embedding_backend' => trim((string)($parsestatus['embedding_backend'] ?? '')),
        'embedding_model' => trim((string)($parsestatus['embedding_model'] ?? '')),
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
 * Remove directory recursively.
 *
 * @param string $path
 * @return void
 */
function local_chatbot_delete_dir_recursive(string $path): void {
    if ($path === '' || !is_dir($path)) {
        return;
    }
    foreach (scandir($path) as $name) {
        if ($name === '.' || $name === '..') {
            continue;
        }
        $child = $path . DIRECTORY_SEPARATOR . $name;
        if (is_dir($child)) {
            local_chatbot_delete_dir_recursive($child);
        } else {
            @unlink($child);
        }
    }
    @rmdir($path);
}

/**
 * Remove cached vector index directories and manifests from data directory.
 *
 * @return void
 */
function local_chatbot_clear_data_dir_indexes(): void {
    local_chatbot_ensure_data_dir();
    $datadir = local_chatbot_get_data_path();
    foreach (scandir($datadir) as $name) {
        if ($name === '.' || $name === '..') {
            continue;
        }
        $path = $datadir . DIRECTORY_SEPARATOR . $name;
        if (is_dir($path) && strpos($name, '.rag_chroma') === 0) {
            local_chatbot_delete_dir_recursive($path);
            continue;
        }
        if (is_file($path) && strpos($name, '.rag_index_manifest') === 0) {
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
    $embedbackend = isset($tracecontext['embed_backend']) ? trim((string)$tracecontext['embed_backend']) : '';

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
    $previousembedbackend = getenv('EMBED_BACKEND');
    if ($embedbackend !== '') {
        @putenv('EMBED_BACKEND=' . $embedbackend);
    }
    exec($cmd, $output, $code);
    if ($embedbackend !== '') {
        if ($previousembedbackend === false) {
            @putenv('EMBED_BACKEND');
        } else {
            @putenv('EMBED_BACKEND=' . $previousembedbackend);
        }
    }
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
    $evalmode = !empty($tracecontext['eval_mode']);
    $questionid = isset($tracecontext['question_id']) ? trim((string)$tracecontext['question_id']) : '';
    $runid = isset($tracecontext['run_id']) ? (int)$tracecontext['run_id'] : 0;
    $evalmodename = isset($tracecontext['eval_mode_name']) ? trim((string)$tracecontext['eval_mode_name']) : '';
    $rawresultspath = isset($tracecontext['raw_results_path']) ? trim((string)$tracecontext['raw_results_path']) : '';
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
    if ($evalmode) {
        $runnerargs[] = '--eval-mode';
        if ($questionid !== '') {
            $runnerargs[] = '--question-id';
            $runnerargs[] = $questionid;
        }
        if ($runid > 0) {
            $runnerargs[] = '--run-id';
            $runnerargs[] = (string)$runid;
        }
        if ($rawresultspath !== '') {
            $runnerargs[] = '--raw-results-path';
            $runnerargs[] = $rawresultspath;
        }
        if ($evalmodename !== '') {
            $runnerargs[] = '--eval-mode-name';
            $runnerargs[] = $evalmodename;
        }
    }

    $payload = local_chatbot_run_runner_command($runnerargs, $tracecontext);
    if (!array_key_exists('answer', $payload)) {
        throw new Exception('Invalid runner response: missing answer field.');
    }

    return [
        'answer' => (string)$payload['answer'],
        'sources' => isset($payload['sources']) && is_array($payload['sources']) ? $payload['sources'] : [],
        'mode' => trim((string)($payload['mode'] ?? '')),
        'question_id' => trim((string)($payload['question_id'] ?? '')),
        'run_id' => max(0, (int)($payload['run_id'] ?? 0)),
        'model_name' => trim((string)($payload['model_name'] ?? '')),
        'embedding_backend' => trim((string)($payload['embedding_backend'] ?? '')),
        'embedding_model_name' => trim((string)($payload['embedding_model_name'] ?? '')),
        'latency_total' => max(0.0, (float)($payload['latency_total'] ?? 0.0)),
        'latency_retrieval' => max(0.0, (float)($payload['latency_retrieval'] ?? 0.0)),
        'latency_generation' => max(0.0, (float)($payload['latency_generation'] ?? 0.0)),
        'retrieved_context_count' => max(0, (int)($payload['retrieved_context_count'] ?? 0)),
        'status' => core_text::strtolower(trim((string)($payload['status'] ?? 'success'))),
        'error_message' => isset($payload['error_message']) ? trim((string)$payload['error_message']) : '',
    ];
}

/**
 * Run pre-parse/index warmup for current chatbot data directory.
 *
 * @return array{ok:bool,preparsed:bool,rebuilt:bool,sources:int,embedding_backend:string}
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
        'embedding_backend' => trim((string)($payload['embedding_backend'] ?? '')),
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
    $evalmode = !empty($tracecontext['eval_mode']);

    if (
        !$evalmode &&
        !$issimplegreeting &&
        $islongprompt &&
        local_chatbot_is_generic_fallback_answer((string)$result['answer'])
    ) {
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
        local_chatbot_clear_data_dir_indexes();
        local_chatbot_write_material_context_state('none');
        return [];
    }

    local_chatbot_clear_data_dir_documents();
    local_chatbot_clear_data_dir_indexes();
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
    $files = local_chatbot_list_uploaded_files();
    if (empty($files)) {
        local_chatbot_write_material_context_state('none');
        return [];
    }

    $course = get_course($courseid);
    $coursename = '';
    if ($course) {
        $coursename = trim((string)($course->fullname ?? ''));
        if ($coursename === '') {
            $coursename = trim((string)($course->shortname ?? ''));
        }
    }
    local_chatbot_write_material_context_state('topic', [
        'course_id' => $courseid,
        'course_name' => $coursename,
        'topic' => trim($topic),
    ]);

    return $files;
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
