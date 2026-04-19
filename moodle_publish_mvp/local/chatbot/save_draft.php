<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

/**
 * Save draft endpoint for LLM assignment output.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/locallib.php');

use local_chatbot\service\draft_repository;
use local_chatbot\service\draft_validator;
use local_chatbot\service\markdown_draft_parser;
use local_chatbot\service\weight_ui_service;

/**
 * Normalize one optional bucket type.
 *
 * @param string $value
 * @return string
 */
function local_chatbot_normalize_weight_bucket_type(string $value): string {
    $normalized = strtolower(trim($value));
    $allowed = ['individual', 'group', 'practice', 'quiz', 'uts', 'uas'];
    return in_array($normalized, $allowed, true) ? $normalized : '';
}

/**
 * Normalize source value for mapping.
 *
 * @param string $value
 * @return string
 */
function local_chatbot_normalize_weight_source(string $value): string {
    $normalized = strtolower(trim($value));
    if ($normalized === weight_ui_service::SOURCE_TEACHER) {
        return weight_ui_service::SOURCE_TEACHER;
    }
    return weight_ui_service::SOURCE_LLM;
}

require_login();
require_sesskey();

$courseid = required_param('courseid', PARAM_INT);
$course = get_course($courseid);
require_login($course);
$contentmode = optional_param('content_mode', 'assignment', PARAM_ALPHANUMEXT);
$normalizedmode = strtolower(trim((string)$contentmode));
if (!in_array($normalizedmode, ['assignment', 'practice'], true)) {
    $normalizedmode = 'assignment';
}

$context = context_course::instance($courseid);
if ($normalizedmode === 'practice') {
    if (!local_chatbot_user_can_access_course_materials($courseid, (int)$USER->id)) {
        throw new required_capability_exception($context, 'moodle/course:view', 'nopermissions', '');
    }
} else {
    require_capability('local/chatbot:managedrafts', $context);
}

$title = optional_param('title', '', PARAM_TEXT);
$topic = optional_param('topic', '', PARAM_TEXT);
$assignmenttype = optional_param('assignment_type', 'multiple_choice', PARAM_ALPHANUMEXT);
$assignmenttypelabel = optional_param('assignment_type_label', '', PARAM_TEXT);
$weightbuckettype = optional_param('weight_bucket_type', '', PARAM_ALPHANUMEXT);
$activityweightlabel = optional_param('activity_weight_label', '', PARAM_TEXT);
$activityweightpercentraw = optional_param('activity_weight_percent', '', PARAM_RAW_TRIMMED);
$weightsource = optional_param('weight_source', '', PARAM_ALPHANUMEXT);
$questioncount = optional_param('question_count', 0, PARAM_INT);
$essayautogradeenabled = optional_param('essay_autograde_enabled', 0, PARAM_BOOL);
$rawdraftjson = optional_param('draft_json', '', PARAM_RAW);
$rawdrafttext = optional_param('draft_text', '', PARAM_RAW);

$response = ['success' => false, 'message' => ''];
$repository = new draft_repository();
$validator = new draft_validator();
$parser = new markdown_draft_parser();

try {
    $payload = null;
    if (trim($rawdraftjson) !== '') {
        $payload = json_decode($rawdraftjson, true);
        if (!is_array($payload)) {
            throw new moodle_exception('invaliddraftjson', 'local_chatbot');
        }
    } elseif (trim($rawdrafttext) !== '') {
        $payload = $parser->parse($rawdrafttext);
    } else {
        throw new moodle_exception('missingdraftpayload', 'local_chatbot');
    }

    if (trim($title) !== '') {
        $payload['assignment_title'] = $title;
    }
    if (trim($topic) !== '') {
        $payload['topic'] = trim($topic);
    }
    $payload['content_mode'] = $normalizedmode;
    $normalizedtype = strtolower(trim(str_replace('_', '-', (string)$assignmenttype)));
    if (trim($assignmenttypelabel) !== '') {
        $payload['assignment_type_label'] = trim($assignmenttypelabel);
    }

    if ($normalizedmode === 'assignment') {
        $normalizedbuckettype = local_chatbot_normalize_weight_bucket_type((string)$weightbuckettype);
        if ($normalizedbuckettype !== '') {
            $payload['weight_bucket_type'] = $normalizedbuckettype;
        }

        $normalizedweightlabel = weight_ui_service::normalize_weight_label(
            trim((string)$activityweightlabel) !== '' ? (string)$activityweightlabel : 'medium'
        );
        $weightpercent = weight_ui_service::weight_percent_from_label($normalizedweightlabel);
        if (trim((string)$activityweightpercentraw) !== '' && is_numeric((string)$activityweightpercentraw)) {
            $weightpercent = weight_ui_service::clamp_weight_percent((float)$activityweightpercentraw);
        }
        $payload['activity_weight_label'] = $normalizedweightlabel;
        $payload['activity_weight_percent'] = $weightpercent;
        $payload['weight_source'] = local_chatbot_normalize_weight_source(
            trim((string)$weightsource) !== '' ? (string)$weightsource : weight_ui_service::SOURCE_LLM
        );
    }

    $payload['essay_autograde_enabled'] = (
        $normalizedmode === 'assignment' &&
        $normalizedtype === 'essay' &&
        !empty($essayautogradeenabled)
    ) ? 1 : 0;
    if ($questioncount > 10) {
        $questioncount = 10;
    }
    if ($questioncount <= 0 && !empty($payload['questions']) && is_array($payload['questions'])) {
        $questioncount = count($payload['questions']);
    }
    if ($questioncount > 10) {
        $questioncount = 10;
        if (isset($payload['questions']) && is_array($payload['questions'])) {
            $payload['questions'] = array_slice($payload['questions'], 0, 10);
        }
        if (isset($payload['answer_key']) && is_array($payload['answer_key'])) {
            $payload['answer_key'] = array_slice($payload['answer_key'], 0, 10, true);
        }
    }

    $validator->validate_payload($payload, $questioncount, $assignmenttype);

    $draftid = $repository->create(
        $courseid,
        (int)$USER->id,
        (string)$payload['assignment_title'],
        $assignmenttype,
        $questioncount,
        $payload,
        'draft'
    );

    $response = [
        'success' => true,
        'message' => get_string('savedraftsuccess', 'local_chatbot'),
        'draftid' => $draftid,
    ];
} catch (Throwable $exception) {
    $response['message'] = $exception->getMessage();
}

@header('Content-Type: application/json; charset=utf-8');
echo json_encode($response);
