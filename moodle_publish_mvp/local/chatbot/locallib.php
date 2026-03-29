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
        $file = 'moodle_rag_runner.py';
    }
    return $file;
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
 * @return array
 */
function local_chatbot_run_rag_once(string $question): array {
    $python = local_chatbot_get_python_path();
    $projectpath = local_chatbot_get_project_path();
    $runner = $projectpath . DIRECTORY_SEPARATOR . local_chatbot_get_runner_file();
    $datadir = local_chatbot_get_data_path();

    if (!is_file($python)) {
        throw new Exception('Python executable not found: ' . $python);
    }
    if (!is_file($runner)) {
        throw new Exception('Runner script not found: ' . $runner);
    }

    $questionb64 = base64_encode($question);
    $cmd = local_chatbot_quote_arg($python) . ' ' .
        local_chatbot_quote_arg($runner) . ' --data-dir ' .
        local_chatbot_quote_arg($datadir) . ' --query-b64 ' .
        local_chatbot_quote_arg($questionb64) . ' 2>&1';

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
    $result = local_chatbot_run_rag_once($question);
    $normalizedquestion = core_text::strtolower(trim($question));
    $islongprompt = core_text::strlen(trim($question)) >= 80;
    $issimplegreeting = in_array($normalizedquestion, ['hi', 'hello', 'halo', 'hey'], true);

    if (!$issimplegreeting && $islongprompt && local_chatbot_is_generic_fallback_answer((string)$result['answer'])) {
        $result = local_chatbot_run_rag_once($question);
    }

    return $result;
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
 * Lists class topics (section names) for a course the user can manage.
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

    $context = context_course::instance($courseid);
    if (!is_siteadmin($userid) && !has_capability('moodle/course:update', $context, $userid)) {
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
 * Lists PDF resource files available in a course the user can manage.
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

    $context = context_course::instance($courseid);
    if (!is_siteadmin($userid) && !has_capability('moodle/course:update', $context, $userid)) {
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

    $context = context_course::instance($courseid);
    if (!is_siteadmin($userid) && !has_capability('moodle/course:update', $context, $userid)) {
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
 * Resolve a manageable course id from course label/shortname for current user.
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
        $context = context_course::instance((int)$course->id, IGNORE_MISSING);
        if (!$context) {
            continue;
        }
        if (is_siteadmin($userid) || has_capability('moodle/course:update', $context, $userid)) {
            return (int)$course->id;
        }
    }
    return 0;
}
