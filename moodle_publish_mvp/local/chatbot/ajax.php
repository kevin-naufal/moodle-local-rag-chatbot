<?php
define('AJAX_SCRIPT', true);

require_once(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/locallib.php');

use local_chatbot\service\essay_autograder;
use local_chatbot\service\essay_grade_repository;

require_login();
require_sesskey();

$context = context_system::instance();
require_capability('local/chatbot:view', $context);

header('Content-Type: application/json');
$localchatbotresponded = false;

/**
 * Emit JSON response with robust UTF-8 handling.
 *
 * @param array $payload
 * @param int $statuscode
 * @return void
 */
function local_chatbot_emit_json(array $payload, int $statuscode = 200): void {
    global $localchatbotresponded;
    if (!headers_sent()) {
        http_response_code($statuscode);
    }
    $json = json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_INVALID_UTF8_SUBSTITUTE);
    if ($json === false) {
        $json = '{"ok":false,"error":"JSON encoding failed."}';
    }
    echo $json;
    $localchatbotresponded = true;
}

/**
 * Extend request execution time for long-running LLM calls.
 *
 * @param int $seconds
 * @return void
 */
function local_chatbot_extend_execution_time(int $seconds = 300): void {
    if ($seconds < 1) {
        return;
    }
    if (function_exists('set_time_limit')) {
        @set_time_limit($seconds);
    }
    @ini_set('max_execution_time', (string)$seconds);
}

register_shutdown_function(static function(): void {
    global $localchatbotresponded;
    if ($localchatbotresponded) {
        return;
    }
    $error = error_get_last();
    if ($error === null) {
        return;
    }

    $fataltypes = [E_ERROR, E_PARSE, E_CORE_ERROR, E_COMPILE_ERROR, E_USER_ERROR];
    if (!in_array((int)$error['type'], $fataltypes, true)) {
        return;
    }

    while (ob_get_level() > 0) {
        @ob_end_clean();
    }
    local_chatbot_emit_json([
        'ok' => false,
        'error' => 'Fatal server error: ' . (string)($error['message'] ?? 'Unknown fatal error'),
    ], 500);
});

try {
    $action = required_param('action', PARAM_ALPHAEXT);
    if ($action === 'list_files') {
        echo json_encode([
            'ok' => true,
            'files' => local_chatbot_list_uploaded_files(),
        ]);
        exit;
    }

    if ($action === 'upload') {
        if (!local_chatbot_user_is_teacher_like((int)$USER->id) && !is_siteadmin()) {
            echo json_encode(['ok' => false, 'error' => 'Only teachers can upload files.']);
            exit;
        }

        local_chatbot_ensure_data_dir();
        $datadir = local_chatbot_get_data_path();
        $saved = 0;

        if (empty($_FILES['documents']) || !isset($_FILES['documents']['name'])) {
            echo json_encode(['ok' => false, 'error' => 'No files selected']);
            exit;
        }

        $names = $_FILES['documents']['name'];
        $tmps = $_FILES['documents']['tmp_name'];
        $errors = $_FILES['documents']['error'];

        foreach ($names as $i => $name) {
            if ($errors[$i] !== UPLOAD_ERR_OK) {
                continue;
            }

            $basename = clean_param($name, PARAM_FILE);
            $ext = strtolower(pathinfo($basename, PATHINFO_EXTENSION));
            if ($ext !== 'pdf' && $ext !== 'txt') {
                continue;
            }

            $target = $datadir . DIRECTORY_SEPARATOR . $basename;
            if (move_uploaded_file($tmps[$i], $target)) {
                $saved++;
            }
        }

        echo json_encode([
            'ok' => true,
            'saved' => $saved,
            'files' => local_chatbot_list_uploaded_files(),
        ]);
        exit;
    }

    if ($action === 'chat') {
        local_chatbot_extend_execution_time(300);
        $requeststarted = microtime(true);
        $question = required_param('question', PARAM_RAW_TRIMMED);
        $courseid = optional_param('courseid', 0, PARAM_INT);
        $topic = optional_param('topic', '', PARAM_RAW_TRIMMED);
        $generationmode = trim(core_text::strtolower(optional_param('generation_mode', '', PARAM_ALPHANUMEXT)));
        $pagestart = optional_param('page_start', 0, PARAM_INT);
        $pageend = optional_param('page_end', 0, PARAM_INT);
        $requestidraw = optional_param('request_id', '', PARAM_RAW_TRIMMED);
        $requestid = trim((string)preg_replace('/[^a-zA-Z0-9._:-]/', '', $requestidraw));
        if ($requestid === '') {
            $requestid = local_chatbot_generate_request_id();
        }
        $questionnumber = optional_param('question_number', 0, PARAM_INT);
        $generationattempt = optional_param('generation_attempt', 0, PARAM_INT);
        $tracecontext = [
            'request_id' => $requestid,
            'question_number' => $questionnumber,
            'attempt' => $generationattempt,
            'page_start' => $pagestart,
            'page_end' => $pageend,
        ];
        $questionpreview = local_chatbot_trace_truncate_text($question, 2500);
        local_chatbot_trace_log('chat_request_received', [
            'request_id' => $requestid,
            'question_number' => $questionnumber,
            'attempt' => $generationattempt,
            'user_id' => (int)$USER->id,
            'course_id' => $courseid,
            'topic' => trim((string)$topic),
            'generation_mode' => $generationmode,
            'page_start' => $pagestart,
            'page_end' => $pageend,
            'question_chars' => core_text::strlen($question),
            'question_text' => $questionpreview['text'],
            'question_truncated' => !empty($questionpreview['truncated']),
        ]);
        $isstructuredgeneration = local_chatbot_is_structured_generation_prompt($question);
        $topicstatus = 'topic_context_unresolved';
        $activetopic = null;
        $taskgroup = null;
        $chatgroup = null;

        if ($isstructuredgeneration && ($courseid <= 0 || trim($topic) === '')) {
            global $SESSION;
            $lasttaskcontext = isset($SESSION->local_chatbot_last_task_context) &&
                is_array($SESSION->local_chatbot_last_task_context)
                ? $SESSION->local_chatbot_last_task_context
                : null;
            if ($lasttaskcontext !== null &&
                (int)($lasttaskcontext['userid'] ?? 0) === (int)$USER->id &&
                (int)($lasttaskcontext['courseid'] ?? 0) > 0 &&
                trim((string)($lasttaskcontext['topic'] ?? '')) !== '') {
                $courseid = (int)$lasttaskcontext['courseid'];
                $topic = trim((string)$lasttaskcontext['topic']);
            }
        }

        if ($courseid > 0 && trim($topic) !== '') {
            $topiccontext = local_chatbot_resolve_course_topic_name($courseid, (int)$USER->id, $topic);
            $topicstatus = (string)($topiccontext['status'] ?? $topicstatus);
            $activetopic = $topiccontext['active_topic'] === null ? null : (string)$topiccontext['active_topic'];
        }

        $preparedquestion = trim($question);
        if ($isstructuredgeneration) {
            $studentpractice = ($generationmode === 'practice') && local_chatbot_user_is_student_like((int)$USER->id);
            if ($studentpractice) {
                $defaultgroup = 'mid';
                $taskgroup = $defaultgroup;
                if ($courseid > 0 && trim($topic) !== '') {
                    $taskcontext = local_chatbot_resolve_active_topic_context(
                        (int)$USER->id,
                        $courseid,
                        $topic,
                        $defaultgroup
                    );
                    $taskgroup = (string)($taskcontext['group'] ?? $defaultgroup);
                    $topicstatus = (string)($taskcontext['status'] ?? $topicstatus);
                    $activetopic = $taskcontext['active_topic'] === null ? null : (string)$taskcontext['active_topic'];
                }
                $taskmodifier = local_chatbot_build_task_generation_level_modifier(
                    $taskgroup,
                    $activetopic === null ? '' : $activetopic
                );
                $preparedquestion = $taskmodifier . "\n" . $preparedquestion;
            } else {
                // Assignment generation and teacher practice generation must not be influenced by learner mastery.
                $taskgroup = 'teacher_task';
            }
        } else {
            $defaultgroup = 'mid';
            $chatgroup = $defaultgroup;
            $hasmaterialcontext = ($courseid > 0 && trim($topic) !== '');
            if ($hasmaterialcontext) {
                $chatcontext = local_chatbot_resolve_active_topic_context(
                    (int)$USER->id,
                    $courseid,
                    $topic,
                    $defaultgroup
                );
                $chatgroup = (string)($chatcontext['group'] ?? $defaultgroup);
                $topicstatus = (string)($chatcontext['status'] ?? $topicstatus);
                $activetopic = $chatcontext['active_topic'] === null ? null : (string)$chatcontext['active_topic'];
            } else {
                $topicstatus = 'general_mode_without_material_context';
                $activetopic = null;
            }
            $chatmodifier = local_chatbot_build_chatbot_language_level_modifier(
                $chatgroup,
                $activetopic === null ? '' : $activetopic
            );
            $preparedquestion = $chatmodifier . "\nQuestion: " . $preparedquestion;
        }

        if (!$isstructuredgeneration && $courseid <= 0 && trim((string)$topic) === '') {
            $result = local_chatbot_run_llm_general($preparedquestion, false, $tracecontext);
            $result['sources'] = [];
        } else {
            $result = local_chatbot_run_rag($preparedquestion, $tracecontext);
        }
        $lowjargonmeta = null;
        if (!$isstructuredgeneration) {
            $normalizedanswer = local_chatbot_normalize_chat_answer((string)$result['answer']);
            if ((string)$chatgroup === 'low') {
                $enforced = local_chatbot_enforce_low_jargon($normalizedanswer);
                $result['answer'] = (string)($enforced['answer'] ?? $normalizedanswer);
                $lowjargonmeta = [
                    'enforced' => !empty($enforced['enforced']),
                    'rewritten' => !empty($enforced['rewritten']),
                    'unexplained_terms' => isset($enforced['unexplained_terms']) && is_array($enforced['unexplained_terms'])
                        ? array_values($enforced['unexplained_terms'])
                        : [],
                ];
            } else {
                $result['answer'] = $normalizedanswer;
            }
        }
        $durationms = (int)round((microtime(true) - $requeststarted) * 1000);
        $answerpreview = local_chatbot_trace_truncate_text((string)$result['answer'], 3000);
        local_chatbot_trace_log('chat_request_success', [
            'request_id' => $requestid,
            'question_number' => $questionnumber,
            'attempt' => $generationattempt,
            'duration_ms' => $durationms,
            'sources_count' => isset($result['sources']) && is_array($result['sources']) ? count($result['sources']) : 0,
            'answer_chars' => isset($result['answer']) ? core_text::strlen((string)$result['answer']) : 0,
            'answer_text' => $answerpreview['text'],
            'answer_truncated' => !empty($answerpreview['truncated']),
            'llm_group' => $isstructuredgeneration ? $taskgroup : $chatgroup,
            'topic_status' => $topicstatus,
            'generation_mode' => $generationmode,
            'page_start' => $pagestart,
            'page_end' => $pageend,
        ]);
        local_chatbot_emit_json([
            'ok' => true,
            'answer' => $result['answer'],
            'sources' => $result['sources'],
            'request_id' => $requestid,
            'llm_group' => $isstructuredgeneration ? $taskgroup : $chatgroup,
            'topic_status' => $topicstatus,
            'active_topic' => $activetopic,
            'low_jargon_meta' => $lowjargonmeta,
        ]);
        exit;
    }

    if ($action === 'grade_essay') {
        $courseid = required_param('courseid', PARAM_INT);
        $course = get_course($courseid);
        require_login($course);

        $coursecontext = context_course::instance($courseid);
        require_capability('local/chatbot:managedrafts', $coursecontext);

        $questiontext = required_param('question_text', PARAM_RAW_TRIMMED);
        $expectedkeypoints = required_param('expected_key_points', PARAM_RAW_TRIMMED);
        $studentanswer = optional_param('student_answer', '', PARAM_RAW_TRIMMED);
        $questionnumber = optional_param('question_number', 1, PARAM_INT);
        $rubricid = optional_param('rubric_id', 'essay_default_v1', PARAM_ALPHANUMEXT);
        $studentid = optional_param('student_id', 0, PARAM_INT);
        $saveresult = optional_param('save_result', 1, PARAM_BOOL);

        $grader = new essay_autograder();
        $grading = $grader->grade([
            'question_text' => $questiontext,
            'expected_key_points' => $expectedkeypoints,
            'student_answer' => $studentanswer,
            'question_number' => $questionnumber,
            'rubric_id' => $rubricid,
        ]);

        $gradeid = 0;
        if ($saveresult) {
            global $DB;
            if ($studentid > 0 && !$DB->record_exists('user', ['id' => $studentid])) {
                throw new invalid_parameter_exception('Invalid student_id.');
            }
            $repository = new essay_grade_repository();
            $gradeid = $repository->create(
                $courseid,
                (int)$USER->id,
                max(0, $studentid),
                $questionnumber,
                $rubricid,
                $questiontext,
                $expectedkeypoints,
                $studentanswer,
                $grading
            );
        }

        echo json_encode([
            'ok' => true,
            'grading' => $grading,
            'saved' => (bool)$saveresult,
            'gradeid' => $gradeid,
        ]);
        exit;
    }

    if ($action === 'set_material_context') {
        $courseid = required_param('courseid', PARAM_INT);
        $coursename = optional_param('course_name', '', PARAM_TEXT);
        if ($courseid <= 0 && $coursename !== '') {
            $courseid = local_chatbot_resolve_courseid_for_teacher($coursename, (int)$USER->id);
        }
        if ($courseid <= 0) {
            echo json_encode(['ok' => false, 'error' => 'Invalid course']);
            exit;
        }

        if (!local_chatbot_user_can_access_course_materials($courseid, (int)$USER->id)) {
            echo json_encode(['ok' => false, 'error' => 'You cannot access this course material.']);
            exit;
        }

        $topic = optional_param('topic', '', PARAM_RAW_TRIMMED);
        $trimmedtopic = trim((string)$topic);
        $files = local_chatbot_sync_course_topic_materials_to_data($courseid, (int)$USER->id, $topic);
        $masterycontext = null;
        if ($courseid > 0 && $trimmedtopic !== '') {
            $resolved = local_chatbot_resolve_active_topic_context(
                (int)$USER->id,
                (int)$courseid,
                $trimmedtopic,
                'mid'
            );
            $masterycontext = [
                'status' => (string)($resolved['status'] ?? 'topic_not_resolved'),
                'active_topic' => isset($resolved['active_topic']) ? (string)($resolved['active_topic'] ?? '') : '',
                'mastery' => isset($resolved['mastery']) && $resolved['mastery'] !== null
                    ? (float)$resolved['mastery']
                    : null,
                'group' => (string)($resolved['group'] ?? 'mid'),
                'fallback_group' => isset($resolved['fallback_group']) && $resolved['fallback_group'] !== null
                    ? (string)$resolved['fallback_group']
                    : null,
            ];
        }
        global $SESSION;
        $SESSION->local_chatbot_last_task_context = [
            'userid' => (int)$USER->id,
            'courseid' => (int)$courseid,
            'topic' => $trimmedtopic,
        ];
        echo json_encode([
            'ok' => true,
            'files' => $files,
            'mastery_context' => $masterycontext,
            'parse_status' => local_chatbot_get_current_material_parse_status(),
        ]);
        exit;
    }

    if ($action === 'parse_status') {
        echo json_encode([
            'ok' => true,
            'parse_status' => local_chatbot_get_current_material_parse_status(),
        ]);
        exit;
    }

    if ($action === 'render_markdown') {
        $text = required_param('text', PARAM_RAW);
        $html = format_text(
            $text,
            FORMAT_MARKDOWN,
            [
                'context' => $context,
                'trusted' => false,
                'noclean' => false,
                'filter' => true,
            ]
        );
        echo json_encode([
            'ok' => true,
            'html' => $html,
        ]);
        exit;
    }

    if ($action === 'course_topics') {
        $courseid = required_param('courseid', PARAM_INT);
        $topics = local_chatbot_list_course_topics($courseid, (int)$USER->id);
        echo json_encode([
            'ok' => true,
            'topics' => $topics,
        ]);
        exit;
    }

    if ($action === 'course_pdfs') {
        $courseid = required_param('courseid', PARAM_INT);
        $coursename = optional_param('course_name', '', PARAM_TEXT);
        if ($courseid <= 0 && $coursename !== '') {
            $courseid = local_chatbot_resolve_courseid_for_teacher($coursename, (int)$USER->id);
        }
        $topic = optional_param('topic', '', PARAM_RAW_TRIMMED);
        $pdfs = local_chatbot_list_course_pdfs($courseid, (int)$USER->id, $topic);
        echo json_encode([
            'ok' => true,
            'pdfs' => $pdfs,
        ]);
        exit;
    }

    if ($action === 'topic_mastery') {
        $courseid = required_param('courseid', PARAM_INT);
        $coursename = optional_param('course_name', '', PARAM_TEXT);
        if ($courseid <= 0 && $coursename !== '') {
            $courseid = local_chatbot_resolve_courseid_for_teacher($coursename, (int)$USER->id);
        }
        if ($courseid <= 0) {
            echo json_encode(['ok' => false, 'error' => 'Invalid course']);
            exit;
        }
        if (!local_chatbot_user_can_access_course_materials($courseid, (int)$USER->id)) {
            echo json_encode(['ok' => false, 'error' => 'You cannot access this course material.']);
            exit;
        }

        $topic = optional_param('topic', '', PARAM_RAW_TRIMMED);
        $trimmedtopic = trim((string)$topic);
        if ($trimmedtopic === '') {
            echo json_encode([
                'ok' => true,
                'mastery_context' => null,
            ]);
            exit;
        }

        $resolved = local_chatbot_resolve_active_topic_context(
            (int)$USER->id,
            (int)$courseid,
            $trimmedtopic,
            'mid'
        );
        $masterycontext = [
            'status' => (string)($resolved['status'] ?? 'topic_not_resolved'),
            'active_topic' => isset($resolved['active_topic']) ? (string)($resolved['active_topic'] ?? '') : '',
            'mastery' => isset($resolved['mastery']) && $resolved['mastery'] !== null
                ? (float)$resolved['mastery']
                : null,
            'group' => (string)($resolved['group'] ?? 'mid'),
            'fallback_group' => isset($resolved['fallback_group']) && $resolved['fallback_group'] !== null
                ? (string)$resolved['fallback_group']
                : null,
        ];

        echo json_encode([
            'ok' => true,
            'mastery_context' => $masterycontext,
        ]);
        exit;
    }

    if ($action === 'file_content') {
        $filename = required_param('filename', PARAM_FILE);
        $datadir = local_chatbot_get_data_path();
        $path = $datadir . DIRECTORY_SEPARATOR . $filename;

        if (!is_file($path)) {
            echo json_encode(['ok' => false, 'error' => 'File not found']);
            exit;
        }

        $ext = strtolower(pathinfo($filename, PATHINFO_EXTENSION));
        if ($ext !== 'pdf' && $ext !== 'txt') {
            echo json_encode(['ok' => false, 'error' => 'Unsupported file type']);
            exit;
        }

        if ($ext === 'pdf') {
            $viewurl = (new moodle_url('/local/chatbot/view.php', ['file' => $filename]))->out(false);
            echo json_encode([
                'ok' => true,
                'filename' => $filename,
                'filetype' => 'pdf',
                'viewurl' => $viewurl,
            ]);
            exit;
        }

        $limit = 200000;
        $size = filesize($path);
        $content = file_get_contents($path, false, null, 0, $limit);
        if ($content === false) {
            echo json_encode(['ok' => false, 'error' => 'Failed to read file']);
            exit;
        }

        echo json_encode([
            'ok' => true,
            'filename' => $filename,
            'filetype' => 'txt',
            'content' => $content,
            'truncated' => ($size > $limit),
        ]);
        exit;
    }

    echo json_encode(['ok' => false, 'error' => 'Unsupported action']);
} catch (Throwable $e) {
    $requestidraw = optional_param('request_id', '', PARAM_RAW_TRIMMED);
    $requestid = trim((string)preg_replace('/[^a-zA-Z0-9._:-]/', '', $requestidraw));
    $questionnumber = optional_param('question_number', 0, PARAM_INT);
    $generationattempt = optional_param('generation_attempt', 0, PARAM_INT);
    $generationmode = trim(core_text::strtolower(optional_param('generation_mode', '', PARAM_ALPHANUMEXT)));
    $pagestart = optional_param('page_start', 0, PARAM_INT);
    $pageend = optional_param('page_end', 0, PARAM_INT);
    local_chatbot_trace_log('ajax_request_error', [
        'request_id' => $requestid,
        'question_number' => $questionnumber,
        'attempt' => $generationattempt,
        'generation_mode' => $generationmode,
        'page_start' => $pagestart,
        'page_end' => $pageend,
        'action' => isset($action) ? (string)$action : '',
        'error' => (string)$e->getMessage(),
        'file' => $e->getFile(),
        'line' => $e->getLine(),
    ], 'error');
    local_chatbot_emit_json([
        'ok' => false,
        'error' => (string)$e->getMessage(),
        'request_id' => $requestid !== '' ? $requestid : null,
    ], 500);
}
