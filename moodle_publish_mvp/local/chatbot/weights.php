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

require_once(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/locallib.php');
require_once($CFG->dirroot . '/enrol/locallib.php');

use local_chatbot\service\weight_ui_service;

/**
 * Build list of teacher-manageable courses.
 *
 * @param int $userid
 * @return array<int,\stdClass>
 */
function local_chatbot_weight_manageable_courses(int $userid): array {
    global $DB;

    $courses = [];
    $enrolled = enrol_get_users_courses($userid, true, 'id,fullname,shortname', 'sortorder ASC');
    foreach ($enrolled as $course) {
        if ((int)$course->id === SITEID) {
            continue;
        }
        $coursecontext = context_course::instance((int)$course->id, IGNORE_MISSING);
        if (!$coursecontext) {
            continue;
        }
        if (has_capability('local/chatbot:manageweights', $coursecontext, $userid) ||
            has_capability('moodle/course:update', $coursecontext, $userid) ||
            is_siteadmin($userid)
        ) {
            $courses[(int)$course->id] = $course;
        }
    }

    if (is_siteadmin($userid) && weight_ui_service::tables_ready()) {
        $existing = $DB->get_records_menu(weight_ui_service::TABLE_SCHEME, null, '', 'courseid,courseid');
        foreach ($existing as $courseid) {
            $cid = (int)$courseid;
            if ($cid <= 0 || isset($courses[$cid])) {
                continue;
            }
            $course = $DB->get_record('course', ['id' => $cid], 'id,fullname,shortname', IGNORE_MISSING);
            if ($course && (int)$course->id !== SITEID) {
                $courses[$cid] = $course;
            }
        }
    }

    return $courses;
}

/**
 * Render one number input for percent values.
 *
 * @param string $name
 * @param float $value
 * @return string
 */
function local_chatbot_weight_number_input(string $name, float $value): string {
    return html_writer::empty_tag('input', [
        'type' => 'number',
        'name' => $name,
        'value' => format_float($value, 2),
        'step' => '0.01',
        'min' => '0',
        'max' => '100',
        'class' => 'form-control',
        'style' => 'max-width: 120px;',
    ]);
}

/**
 * Read one posted float value safely.
 *
 * @param string $key
 * @param float $default
 * @return float
 */
function local_chatbot_weight_post_float(string $key, float $default = 0.0): float {
    $raw = trim((string)optional_param($key, (string)$default, PARAM_RAW_TRIMMED));
    if ($raw === '' || !is_numeric($raw)) {
        return $default;
    }
    return (float)$raw;
}

require_login();

$systemcontext = context_system::instance();
require_capability('local/chatbot:view', $systemcontext);

$isteacher = local_chatbot_user_is_teacher_like((int)$USER->id) || is_siteadmin((int)$USER->id);
if (!$isteacher) {
    throw new required_capability_exception($systemcontext, 'local/chatbot:manageweights', 'nopermissions', '');
}

$baseurl = new moodle_url('/local/chatbot/weights.php');
$courseid = optional_param('courseid', 0, PARAM_INT);
$action = optional_param('action', '', PARAM_ALPHAEXT);
$notifications = [];

$PAGE->set_context($systemcontext);
$PAGE->set_pagelayout('report');

if (!weight_ui_service::tables_ready()) {
    $PAGE->set_url($baseurl);
    $PAGE->set_title(get_string('weightsettingstitle', 'local_chatbot'));
    $PAGE->set_heading(get_string('weightsettingstitle', 'local_chatbot'));
    echo $OUTPUT->header();
    echo $OUTPUT->heading(get_string('weightsettingstitle', 'local_chatbot'));
    echo $OUTPUT->notification(get_string('weightstablemissing', 'local_chatbot'), 'error');
    echo $OUTPUT->footer();
    exit;
}

$courses = local_chatbot_weight_manageable_courses((int)$USER->id);
if (empty($courses)) {
    $PAGE->set_url($baseurl);
    $PAGE->set_title(get_string('weightsettingstitle', 'local_chatbot'));
    $PAGE->set_heading(get_string('weightsettingstitle', 'local_chatbot'));
    echo $OUTPUT->header();
    echo $OUTPUT->heading(get_string('weightsettingstitle', 'local_chatbot'));
    echo $OUTPUT->notification(get_string('weightsettingsnocourse', 'local_chatbot'), 'info');
    echo $OUTPUT->footer();
    exit;
}

if ($courseid <= 0 || !array_key_exists($courseid, $courses)) {
    $courseid = (int)array_key_first($courses);
}
$coursecontext = context_course::instance($courseid);
if (!is_siteadmin((int)$USER->id) &&
    !has_capability('local/chatbot:manageweights', $coursecontext, (int)$USER->id) &&
    !has_capability('moodle/course:update', $coursecontext, (int)$USER->id)
) {
    throw new required_capability_exception($coursecontext, 'local/chatbot:manageweights', 'nopermissions', '');
}

$scheme = weight_ui_service::get_or_create_active_scheme($courseid);
$weights = weight_ui_service::get_scheme_weights((int)$scheme->id);
$activities = weight_ui_service::get_course_activities($courseid);
$mapsbycmid = weight_ui_service::get_activity_maps((int)$scheme->id);

if (data_submitted() && confirm_sesskey()) {
    if ($action === 'savescheme') {
        $rawweights = [
            'category' => [
                weight_ui_service::CATEGORY_TASK => local_chatbot_weight_post_float('weight_category_task', 0.0),
                weight_ui_service::CATEGORY_EXAM => local_chatbot_weight_post_float('weight_category_exam', 0.0),
            ],
            'type' => [
                weight_ui_service::CATEGORY_TASK => [
                    weight_ui_service::TYPE_INDIVIDUAL => local_chatbot_weight_post_float('weight_type_task_individual', 0.0),
                    weight_ui_service::TYPE_GROUP => local_chatbot_weight_post_float('weight_type_task_group', 0.0),
                ],
                weight_ui_service::CATEGORY_EXAM => [
                    weight_ui_service::TYPE_PRACTICE => local_chatbot_weight_post_float('weight_type_exam_practice', 0.0),
                    weight_ui_service::TYPE_QUIZ => local_chatbot_weight_post_float('weight_type_exam_quiz', 0.0),
                    weight_ui_service::TYPE_UTS => local_chatbot_weight_post_float('weight_type_exam_uts', 0.0),
                    weight_ui_service::TYPE_UAS => local_chatbot_weight_post_float('weight_type_exam_uas', 0.0),
                ],
            ],
            'source' => [
                weight_ui_service::CATEGORY_TASK => [
                    weight_ui_service::TYPE_INDIVIDUAL => [
                        weight_ui_service::SOURCE_TEACHER =>
                            local_chatbot_weight_post_float('weight_source_task_individual_teacher', 0.0),
                        weight_ui_service::SOURCE_LLM =>
                            local_chatbot_weight_post_float('weight_source_task_individual_llm', 0.0),
                    ],
                    weight_ui_service::TYPE_GROUP => [
                        weight_ui_service::SOURCE_TEACHER =>
                            local_chatbot_weight_post_float('weight_source_task_group_teacher', 0.0),
                        weight_ui_service::SOURCE_LLM =>
                            local_chatbot_weight_post_float('weight_source_task_group_llm', 0.0),
                    ],
                ],
                weight_ui_service::CATEGORY_EXAM => [
                    weight_ui_service::TYPE_PRACTICE => [
                        weight_ui_service::SOURCE_TEACHER =>
                            local_chatbot_weight_post_float('weight_source_exam_practice_teacher', 0.0),
                        weight_ui_service::SOURCE_LLM =>
                            local_chatbot_weight_post_float('weight_source_exam_practice_llm', 0.0),
                    ],
                    weight_ui_service::TYPE_QUIZ => [
                        weight_ui_service::SOURCE_TEACHER =>
                            local_chatbot_weight_post_float('weight_source_exam_quiz_teacher', 0.0),
                        weight_ui_service::SOURCE_LLM =>
                            local_chatbot_weight_post_float('weight_source_exam_quiz_llm', 0.0),
                    ],
                    weight_ui_service::TYPE_UTS => [
                        weight_ui_service::SOURCE_TEACHER =>
                            local_chatbot_weight_post_float('weight_source_exam_uts_teacher', 0.0),
                        weight_ui_service::SOURCE_LLM =>
                            local_chatbot_weight_post_float('weight_source_exam_uts_llm', 0.0),
                    ],
                    weight_ui_service::TYPE_UAS => [
                        weight_ui_service::SOURCE_TEACHER =>
                            local_chatbot_weight_post_float('weight_source_exam_uas_teacher', 0.0),
                        weight_ui_service::SOURCE_LLM =>
                            local_chatbot_weight_post_float('weight_source_exam_uas_llm', 0.0),
                    ],
                ],
            ],
        ];

        $normalized = weight_ui_service::normalize_weights($rawweights);
        $errors = weight_ui_service::validate_weights($normalized);
        if (!empty($errors)) {
            $weights = $normalized;
            foreach ($errors as $error) {
                $notifications[] = ['type' => 'error', 'text' => $error];
            }
        } else {
            weight_ui_service::upsert_rules((int)$scheme->id, $normalized);
            redirect(
                new moodle_url('/local/chatbot/weights.php', ['courseid' => $courseid]),
                get_string('weightssaved', 'local_chatbot'),
                null,
                \core\output\notification::NOTIFY_SUCCESS
            );
        }
    } else if ($action === 'savemap') {
        $categories = optional_param_array('map_category', [], PARAM_ALPHAEXT);
        $types = optional_param_array('map_type', [], PARAM_ALPHAEXT);
        $sources = optional_param_array('map_source', [], PARAM_ALPHAEXT);

        $entries = [];
        $allowedcmids = [];
        foreach ($activities as $activity) {
            $cmid = (int)$activity->cmid;
            $allowedcmids[] = $cmid;
            $currentmap = $mapsbycmid[$cmid] ?? null;
            $itemweight = $currentmap ? (float)$currentmap->item_weight_percent : 100.0;
            $entries[$cmid] = [
                'category' => trim((string)($categories[$cmid] ?? '')),
                'type' => trim((string)($types[$cmid] ?? '')),
                'source' => trim((string)($sources[$cmid] ?? '')),
                'itemweight' => $itemweight,
                'module' => (string)$activity->module,
                'activityname' => (string)$activity->name,
            ];
        }

        $result = weight_ui_service::save_activity_maps((int)$scheme->id, $courseid, $entries, $allowedcmids);
        if (!empty($result['errors'])) {
            foreach ($result['errors'] as $error) {
                $notifications[] = ['type' => 'error', 'text' => $error];
            }
        } else {
            $message = get_string('weightmapsaved', 'local_chatbot', (int)$result['saved']);
            redirect(
                new moodle_url('/local/chatbot/weights.php', ['courseid' => $courseid]),
                $message,
                null,
                \core\output\notification::NOTIFY_SUCCESS
            );
        }
        $mapsbycmid = weight_ui_service::get_activity_maps((int)$scheme->id);
    }
}

$previewrows = weight_ui_service::build_preview_rows($weights, $mapsbycmid);
$pageurl = new moodle_url('/local/chatbot/weights.php', ['courseid' => $courseid]);

$PAGE->set_url($pageurl);
$PAGE->set_title(get_string('weightsettingstitle', 'local_chatbot'));
$PAGE->set_heading(get_string('weightsettingstitle', 'local_chatbot'));

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('weightsettingstitle', 'local_chatbot'));
echo html_writer::tag('p', get_string('weightsettingssubtitle', 'local_chatbot'));
echo $OUTPUT->notification(get_string('weightsautodefaultnotice', 'local_chatbot'), \core\output\notification::NOTIFY_INFO);

$indexurl = new moodle_url('/local/chatbot/index.php');
$reporturl = new moodle_url('/local/chatbot/teacher_report.php');
echo html_writer::div(
    html_writer::link($indexurl, get_string('pluginname', 'local_chatbot'), ['class' => 'btn btn-secondary mr-2']) .
    html_writer::link($reporturl, get_string('teacherreportlink', 'local_chatbot'), ['class' => 'btn btn-outline-secondary']),
    'mb-3'
);

$courseoptions = [];
foreach ($courses as $cid => $course) {
    $courseoptions[(int)$cid] = trim((string)$course->fullname) !== '' ? (string)$course->fullname : (string)$course->shortname;
}
$courseselect = new single_select(new moodle_url('/local/chatbot/weights.php'), 'courseid', $courseoptions, $courseid);
$courseselect->label = get_string('teacherreportfiltercourse', 'local_chatbot');
echo $OUTPUT->render($courseselect);

echo html_writer::tag('h4', get_string('weightsstepsheading', 'local_chatbot'), ['class' => 'mt-3']);
echo html_writer::alist([
    get_string('weightsstep1', 'local_chatbot'),
    get_string('weightsstep2', 'local_chatbot'),
    get_string('weightsstep3', 'local_chatbot'),
    get_string('weightsstep4', 'local_chatbot'),
], ['class' => 'mb-4']);

foreach ($notifications as $notification) {
    $type = $notification['type'] === 'error' ? 'error' : 'info';
    echo $OUTPUT->notification((string)$notification['text'], $type);
}

$categorylabels = [
    weight_ui_service::CATEGORY_TASK => get_string('weightcategorytask', 'local_chatbot'),
    weight_ui_service::CATEGORY_EXAM => get_string('weightcategoryexam', 'local_chatbot'),
];
$typelabels = [
    weight_ui_service::TYPE_INDIVIDUAL => get_string('weighttypeindividual', 'local_chatbot'),
    weight_ui_service::TYPE_GROUP => get_string('weighttypegroup', 'local_chatbot'),
    weight_ui_service::TYPE_PRACTICE => get_string('weighttypepractice', 'local_chatbot'),
    weight_ui_service::TYPE_QUIZ => get_string('weighttypequiz', 'local_chatbot'),
    weight_ui_service::TYPE_UTS => get_string('weighttypeuts', 'local_chatbot'),
    weight_ui_service::TYPE_UAS => get_string('weighttypeuas', 'local_chatbot'),
];
$sourcelabels = [
    weight_ui_service::SOURCE_TEACHER => get_string('weightsourceteacher', 'local_chatbot'),
    weight_ui_service::SOURCE_LLM => get_string('weightsourcellm', 'local_chatbot'),
];

echo html_writer::tag('h4', get_string('weightsrulesheading', 'local_chatbot'), ['class' => 'mt-3']);
echo html_writer::start_tag('form', ['method' => 'post', 'action' => $pageurl->out(false), 'class' => 'mb-4']);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sesskey', 'value' => sesskey()]);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'action', 'value' => 'savescheme']);

$catetable = new html_table();
$catetable->head = [
    get_string('weighttablecategory', 'local_chatbot'),
    get_string('weighttableweightpercent', 'local_chatbot'),
];
$catetable->data = [
    [
        s($categorylabels[weight_ui_service::CATEGORY_TASK]),
        local_chatbot_weight_number_input(
            'weight_category_task',
            (float)$weights['category'][weight_ui_service::CATEGORY_TASK]
        ),
    ],
    [
        s($categorylabels[weight_ui_service::CATEGORY_EXAM]),
        local_chatbot_weight_number_input(
            'weight_category_exam',
            (float)$weights['category'][weight_ui_service::CATEGORY_EXAM]
        ),
    ],
];
echo html_writer::tag('h5', get_string('weightcategoryheading', 'local_chatbot'));
echo html_writer::table($catetable);

$typetable = new html_table();
$typetable->head = [
    get_string('weighttablecategory', 'local_chatbot'),
    get_string('weighttabletype', 'local_chatbot'),
    get_string('weighttableweightpercent', 'local_chatbot'),
];
$typetable->data = [];
foreach (weight_ui_service::types_by_category() as $category => $types) {
    foreach ($types as $type) {
        $fieldname = 'weight_type_' . $category . '_' . $type;
        $typetable->data[] = [
            s($categorylabels[$category]),
            s($typelabels[$type]),
            local_chatbot_weight_number_input($fieldname, (float)$weights['type'][$category][$type]),
        ];
    }
}
echo html_writer::tag('h5', get_string('weighttypeheading', 'local_chatbot'), ['class' => 'mt-3']);
echo html_writer::table($typetable);

$sourcetable = new html_table();
$sourcetable->head = [
    get_string('weighttablecategory', 'local_chatbot'),
    get_string('weighttabletype', 'local_chatbot'),
    get_string('weightsourceteacher', 'local_chatbot') . ' (%)',
    get_string('weightsourcellm', 'local_chatbot') . ' (%)',
];
$sourcetable->data = [];
foreach (weight_ui_service::types_by_category() as $category => $types) {
    foreach ($types as $type) {
        $teachername = 'weight_source_' . $category . '_' . $type . '_teacher';
        $llmname = 'weight_source_' . $category . '_' . $type . '_llm';
        $sourcetable->data[] = [
            s($categorylabels[$category]),
            s($typelabels[$type]),
            local_chatbot_weight_number_input(
                $teachername,
                (float)$weights['source'][$category][$type][weight_ui_service::SOURCE_TEACHER]
            ),
            local_chatbot_weight_number_input(
                $llmname,
                (float)$weights['source'][$category][$type][weight_ui_service::SOURCE_LLM]
            ),
        ];
    }
}
echo html_writer::tag('h5', get_string('weightsourceheading', 'local_chatbot'), ['class' => 'mt-3']);
echo html_writer::table($sourcetable);
echo html_writer::empty_tag('input', [
    'type' => 'submit',
    'class' => 'btn btn-primary',
    'value' => get_string('weightsavescheme', 'local_chatbot'),
]);
echo html_writer::end_tag('form');

echo html_writer::tag('h4', get_string('weightmapheading', 'local_chatbot'), ['class' => 'mt-4']);
if (empty($activities)) {
    echo $OUTPUT->notification(get_string('weightmapemptyactivities', 'local_chatbot'), 'info');
} else {
    $categoryoptions = [
        '' => get_string('choose', 'moodle'),
        weight_ui_service::CATEGORY_TASK => $categorylabels[weight_ui_service::CATEGORY_TASK],
        weight_ui_service::CATEGORY_EXAM => $categorylabels[weight_ui_service::CATEGORY_EXAM],
    ];
    $typeoptions = [
        '' => get_string('choose', 'moodle'),
        weight_ui_service::TYPE_INDIVIDUAL => $categorylabels[weight_ui_service::CATEGORY_TASK] . ' - ' .
            $typelabels[weight_ui_service::TYPE_INDIVIDUAL],
        weight_ui_service::TYPE_GROUP => $categorylabels[weight_ui_service::CATEGORY_TASK] . ' - ' .
            $typelabels[weight_ui_service::TYPE_GROUP],
        weight_ui_service::TYPE_PRACTICE => $categorylabels[weight_ui_service::CATEGORY_EXAM] . ' - ' .
            $typelabels[weight_ui_service::TYPE_PRACTICE],
        weight_ui_service::TYPE_QUIZ => $categorylabels[weight_ui_service::CATEGORY_EXAM] . ' - ' .
            $typelabels[weight_ui_service::TYPE_QUIZ],
        weight_ui_service::TYPE_UTS => $categorylabels[weight_ui_service::CATEGORY_EXAM] . ' - ' .
            $typelabels[weight_ui_service::TYPE_UTS],
        weight_ui_service::TYPE_UAS => $categorylabels[weight_ui_service::CATEGORY_EXAM] . ' - ' .
            $typelabels[weight_ui_service::TYPE_UAS],
    ];
    $sourceoptions = [
        '' => get_string('choose', 'moodle'),
        weight_ui_service::SOURCE_TEACHER => $sourcelabels[weight_ui_service::SOURCE_TEACHER],
        weight_ui_service::SOURCE_LLM => $sourcelabels[weight_ui_service::SOURCE_LLM],
    ];

    echo html_writer::start_tag('form', ['method' => 'post', 'action' => $pageurl->out(false), 'class' => 'mb-4']);
    echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sesskey', 'value' => sesskey()]);
    echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'action', 'value' => 'savemap']);

    $maptable = new html_table();
    $maptable->head = [
        get_string('weighttableactivity', 'local_chatbot'),
        get_string('weighttablemodule', 'local_chatbot'),
        get_string('weighttablecategory', 'local_chatbot'),
        get_string('weighttabletype', 'local_chatbot'),
        get_string('weighttablesource', 'local_chatbot'),
        get_string('weighttableitemweight', 'local_chatbot'),
    ];
    $maptable->data = [];

    foreach ($activities as $activity) {
        $cmid = (int)$activity->cmid;
        $mapped = $mapsbycmid[$cmid] ?? null;

        $defaultcategory = $activity->module === 'assign'
            ? weight_ui_service::CATEGORY_TASK
            : weight_ui_service::CATEGORY_EXAM;
        $defaulttype = $activity->module === 'assign'
            ? weight_ui_service::TYPE_INDIVIDUAL
            : weight_ui_service::TYPE_QUIZ;

        $selectedcategory = $mapped ? (string)$mapped->category : $defaultcategory;
        $selectedtype = $mapped ? (string)$mapped->subtype : $defaulttype;
        $selectedsource = $mapped ? (string)$mapped->source : weight_ui_service::SOURCE_TEACHER;
        $itemweight = $mapped ? (float)$mapped->item_weight_percent : 100.0;

        $activityname = s((string)$activity->name);
        $activitylabel = !empty($activity->url)
            ? html_writer::link((string)$activity->url, $activityname, ['target' => '_blank', 'rel' => 'noopener'])
            : $activityname;

        $maptable->data[] = [
            $activitylabel,
            s(core_text::strtoupper((string)$activity->module)),
            html_writer::select($categoryoptions, 'map_category[' . $cmid . ']', $selectedcategory, false, ['class' => 'custom-select']),
            html_writer::select($typeoptions, 'map_type[' . $cmid . ']', $selectedtype, false, ['class' => 'custom-select']),
            html_writer::select($sourceoptions, 'map_source[' . $cmid . ']', $selectedsource, false, ['class' => 'custom-select']),
            s(format_float($itemweight, 0) . '%'),
        ];
    }

    echo html_writer::table($maptable);
    echo html_writer::empty_tag('input', [
        'type' => 'submit',
        'class' => 'btn btn-primary',
        'value' => get_string('weightsavemap', 'local_chatbot'),
    ]);
    echo html_writer::end_tag('form');
}

echo html_writer::tag('h4', get_string('weightpreviewheading', 'local_chatbot'), ['class' => 'mt-4']);
echo html_writer::tag('p', get_string('weightpreviewformula', 'local_chatbot'));

$previewtable = new html_table();
$previewtable->head = [
    get_string('weighttablecategory', 'local_chatbot'),
    get_string('weighttabletype', 'local_chatbot'),
    get_string('weighttablesource', 'local_chatbot'),
    get_string('weighttableeffective', 'local_chatbot'),
    get_string('weighttablemappedactivities', 'local_chatbot'),
];
$previewtable->data = [];
foreach ($previewrows as $row) {
    $previewtable->data[] = [
        s($categorylabels[(string)$row->category] ?? (string)$row->category),
        s($typelabels[(string)$row->type] ?? (string)$row->type),
        s($sourcelabels[(string)$row->source] ?? (string)$row->source),
        s(format_float((float)$row->effective, 2) . '%'),
        (int)$row->mappedcount,
    ];
}
echo html_writer::table($previewtable);

echo $OUTPUT->footer();
