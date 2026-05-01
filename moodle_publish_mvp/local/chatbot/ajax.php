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

$action = required_param('action', PARAM_ALPHAEXT);

try {
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
        $question = required_param('question', PARAM_RAW_TRIMMED);
        $courseid = optional_param('courseid', 0, PARAM_INT);
        $topic = optional_param('topic', '', PARAM_RAW_TRIMMED);
        $userid = (int)$USER->id;
        $defaultgroup = 'mid';

        $chatgroup = $defaultgroup;
        $topicstatus = 'topic_context_unresolved';
        $activetopic = null;

        if ($courseid > 0 && trim($topic) !== '') {
            $context = local_chatbot_resolve_active_topic_context(
                $userid,
                $courseid,
                $topic,
                $defaultgroup
            );
            $chatgroup = (string)($context['group'] ?? $defaultgroup);
            $topicstatus = (string)($context['status'] ?? $topicstatus);
            $activetopic = $context['active_topic'] === null ? null : (string)$context['active_topic'];
        }

        $modifier = local_chatbot_build_chatbot_level_modifier(
            $chatgroup,
            $activetopic === null ? '' : $activetopic
        );
        $preparedquestion = $modifier . "\nQuestion: " . trim($question);
        $result = local_chatbot_run_rag($preparedquestion);
        if (!local_chatbot_is_structured_generation_prompt($question)) {
            $result['answer'] = local_chatbot_normalize_chat_answer((string)$result['answer']);
        }
        echo json_encode([
            'ok' => true,
            'answer' => $result['answer'],
            'sources' => $result['sources'],
            'llm_group' => $chatgroup,
            'topic_status' => $topicstatus,
            'active_topic' => $activetopic,
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
        $files = local_chatbot_sync_course_topic_materials_to_data($courseid, (int)$USER->id, $topic);
        echo json_encode([
            'ok' => true,
            'files' => $files,
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
    echo json_encode(['ok' => false, 'error' => $e->getMessage()]);
}
