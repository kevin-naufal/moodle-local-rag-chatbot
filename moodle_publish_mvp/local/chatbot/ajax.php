<?php
define('AJAX_SCRIPT', true);

require_once(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/locallib.php');

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
        $result = local_chatbot_run_rag($question);
        echo json_encode([
            'ok' => true,
            'answer' => $result['answer'],
            'sources' => $result['sources'],
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

        $coursecontext = context_course::instance($courseid);
        require_capability('moodle/course:update', $coursecontext);

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
