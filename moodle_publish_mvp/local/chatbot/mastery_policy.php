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

/**
 * Build list of teacher-manageable courses.
 *
 * @param int $userid
 * @return array<int,\stdClass>
 */
function local_chatbot_mastery_manageable_courses(int $userid): array {
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

    if (is_siteadmin($userid) && local_chatbot_learning_tables_ready()) {
        $existing = $DB->get_records_menu('local_chatbot_std_profile', null, '', 'courseid,courseid');
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
 * Read one posted float value safely.
 *
 * @param string $key
 * @param float $default
 * @return float
 */
function local_chatbot_mastery_post_float(string $key, float $default = 0.0): float {
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

$baseurl = new moodle_url('/local/chatbot/mastery_policy.php');
$courseid = optional_param('courseid', 0, PARAM_INT);
$action = optional_param('action', '', PARAM_ALPHAEXT);

$PAGE->set_context($systemcontext);
$PAGE->set_pagelayout('report');

$courses = local_chatbot_mastery_manageable_courses((int)$USER->id);
if (empty($courses)) {
    $PAGE->set_url($baseurl);
    $PAGE->set_title(get_string('masterypolicytitle', 'local_chatbot'));
    $PAGE->set_heading(get_string('masterypolicytitle', 'local_chatbot'));
    echo $OUTPUT->header();
    echo $OUTPUT->heading(get_string('masterypolicytitle', 'local_chatbot'));
    echo $OUTPUT->notification(get_string('masterypolicynocourse', 'local_chatbot'), 'info');
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

$topics = local_chatbot_list_course_topics($courseid, (int)$USER->id);
$topicnames = [];
foreach ($topics as $topic) {
    $name = trim((string)($topic['value'] ?? ''));
    if ($name === '') {
        continue;
    }
    $topicnames[$name] = $name;
}
$topicnames = array_values($topicnames);
$topiclookup = array_fill_keys($topicnames, true);

$policy = local_chatbot_get_course_mastery_policy($courseid);
if (data_submitted() && confirm_sesskey() && $action === 'savepolicy') {
    $defaultminimum = local_chatbot_normalize_mastery_percent(
        local_chatbot_mastery_post_float('default_minimum', (float)$policy['defaultminimum'])
    );
    $requireforexams = optional_param('require_for_exams', 0, PARAM_INT) === 1;

    $postedtopicnames = optional_param_array('topic_name', [], PARAM_TEXT);
    $posteduseflags = optional_param_array('topic_use_default', [], PARAM_INT);
    $postedminimums = optional_param_array('topic_minimum', [], PARAM_RAW_TRIMMED);
    $overrides = [];

    foreach ($postedtopicnames as $index => $topicname) {
        $topicname = trim((string)$topicname);
        if ($topicname === '' || !isset($topiclookup[$topicname])) {
            continue;
        }
        $usedefault = !empty($posteduseflags[$index]);
        if ($usedefault) {
            continue;
        }

        $minimumraw = trim((string)($postedminimums[$index] ?? ''));
        $minimum = is_numeric($minimumraw) ? (float)$minimumraw : $defaultminimum;
        $overrides[$topicname] = local_chatbot_normalize_mastery_percent($minimum);
    }

    local_chatbot_save_course_mastery_policy($courseid, $defaultminimum, $requireforexams, $overrides);
    redirect(
        new moodle_url('/local/chatbot/mastery_policy.php', ['courseid' => $courseid]),
        get_string('masterypolicysaved', 'local_chatbot'),
        null,
        \core\output\notification::NOTIFY_SUCCESS
    );
}

$policy = local_chatbot_get_course_mastery_policy($courseid);
$learningready = local_chatbot_learning_tables_ready();
$profilesbyuser = [];
if ($learningready) {
    $records = $DB->get_records(
        'local_chatbot_std_profile',
        ['courseid' => $courseid],
        '',
        'userid,topic,mastery,timemodified'
    );
    foreach ($records as $record) {
        $userid = (int)$record->userid;
        if ($userid <= 0) {
            continue;
        }
        if (!isset($profilesbyuser[$userid])) {
            $profilesbyuser[$userid] = [];
        }
        $profilesbyuser[$userid][(string)$record->topic] = (object)[
            'mastery' => (float)$record->mastery,
            'timemodified' => (int)$record->timemodified,
        ];
    }
}

$students = get_enrolled_users($coursecontext, 'mod/assign:submit', 0, 'u.id,u.firstname,u.lastname');
if (empty($students)) {
    $students = get_enrolled_users($coursecontext, 'moodle/course:view', 0, 'u.id,u.firstname,u.lastname');
}
usort($students, static function($a, $b): int {
    $aname = trim((string)$a->firstname . ' ' . (string)$a->lastname);
    $bname = trim((string)$b->firstname . ' ' . (string)$b->lastname);
    return strcmp($aname, $bname);
});

$debtsummaryrows = [];
$topiccount = count($topicnames);
foreach ($students as $student) {
    $userid = (int)$student->id;
    $studentname = trim((string)$student->firstname . ' ' . (string)$student->lastname);
    if ($studentname === '') {
        $studentname = get_string('unknownuser', 'moodle');
    }

    $passedcount = 0;
    $debttopics = [];
    $lastupdate = 0;
    $studentprofiles = $profilesbyuser[$userid] ?? [];
    foreach ($topicnames as $topicname) {
        $minimum = local_chatbot_get_course_topic_minimum($topicname, $policy);
        $mastery = isset($studentprofiles[$topicname]) ? (float)$studentprofiles[$topicname]->mastery : 0.0;
        $lastupdate = max($lastupdate, isset($studentprofiles[$topicname]) ? (int)$studentprofiles[$topicname]->timemodified : 0);

        if (local_chatbot_mastery_meets_minimum($mastery, $minimum)) {
            $passedcount++;
            continue;
        }
        $debttopics[] = $topicname;
    }

    $debtcount = max(0, $topiccount - $passedcount);
    $debttopicspreview = '';
    if (!empty($debttopics)) {
        $debttopicspreview = implode(', ', array_slice($debttopics, 0, 3));
        if (count($debttopics) > 3) {
            $debttopicspreview .= ' +' . (count($debttopics) - 3);
        }
    }

    $debtsummaryrows[] = (object)[
        'userid' => $userid,
        'studentname' => $studentname,
        'passedcount' => $passedcount,
        'debtcount' => $debtcount,
        'debttopicspreview' => $debttopicspreview,
        'ready' => $debtcount === 0,
        'lastupdate' => $lastupdate,
    ];
}

usort($debtsummaryrows, static function($a, $b): int {
    $debtcmp = (int)$b->debtcount <=> (int)$a->debtcount;
    if ($debtcmp !== 0) {
        return $debtcmp;
    }
    return strcmp((string)$a->studentname, (string)$b->studentname);
});

$pageurl = new moodle_url('/local/chatbot/mastery_policy.php', ['courseid' => $courseid]);
$PAGE->set_url($pageurl);
$PAGE->set_title(get_string('masterypolicytitle', 'local_chatbot'));
$PAGE->set_heading(get_string('masterypolicytitle', 'local_chatbot'));

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('masterypolicytitle', 'local_chatbot'));
echo html_writer::tag('p', get_string('masterypolicysubtitle', 'local_chatbot'));

$indexurl = new moodle_url('/local/chatbot/index.php');
$weighturl = new moodle_url('/local/chatbot/weights.php', ['courseid' => $courseid]);
echo html_writer::div(
    html_writer::link($indexurl, get_string('pluginname', 'local_chatbot'), ['class' => 'btn btn-secondary mr-2']) .
    html_writer::link($weighturl, get_string('weightsettingslink', 'local_chatbot'), ['class' => 'btn btn-outline-secondary']),
    'mb-3'
);

$courseoptions = [];
foreach ($courses as $cid => $course) {
    $courseoptions[(int)$cid] = trim((string)$course->fullname) !== '' ? (string)$course->fullname : (string)$course->shortname;
}
$courseselect = new single_select(new moodle_url('/local/chatbot/mastery_policy.php'), 'courseid', $courseoptions, $courseid);
$courseselect->label = get_string('teacherreportfiltercourse', 'local_chatbot');
echo $OUTPUT->render($courseselect);

echo html_writer::tag('h4', get_string('masterypolicysectionrules', 'local_chatbot'), ['class' => 'mt-3']);
echo html_writer::start_tag('form', ['method' => 'post', 'action' => $pageurl->out(false), 'class' => 'mb-4']);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sesskey', 'value' => sesskey()]);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'action', 'value' => 'savepolicy']);

$defaultrow = html_writer::div(
    html_writer::tag('label', get_string('masterypolicydefaultminimum', 'local_chatbot'), ['for' => 'local-chatbot-default-minimum']) .
    html_writer::empty_tag('input', [
        'type' => 'number',
        'id' => 'local-chatbot-default-minimum',
        'name' => 'default_minimum',
        'value' => format_float((float)$policy['defaultminimum'], 2),
        'min' => '0',
        'max' => '100',
        'step' => '0.01',
        'class' => 'form-control',
        'style' => 'max-width: 160px;',
    ]),
    'mb-3'
);
echo $defaultrow;

echo html_writer::div(
    html_writer::checkbox(
        'require_for_exams',
        1,
        !empty($policy['requireforexams']),
        get_string('masterypolicyrequireexam', 'local_chatbot')
    ),
    'mb-3'
);

if (empty($topicnames)) {
    echo $OUTPUT->notification(get_string('masterypolicynotopics', 'local_chatbot'), 'info');
} else {
    $topictable = new html_table();
    $topictable->head = [
        get_string('dashboardtabletopic', 'local_chatbot'),
        get_string('masterypolicyusedefault', 'local_chatbot'),
        get_string('masterypolicyoverride', 'local_chatbot'),
        get_string('masterypolicyeffective', 'local_chatbot'),
    ];
    $topictable->data = [];

    foreach ($topicnames as $index => $topicname) {
        $hasoverride = array_key_exists($topicname, $policy['overrides']);
        $topicminimum = $hasoverride
            ? (float)$policy['overrides'][$topicname]
            : (float)$policy['defaultminimum'];

        $topictable->data[] = [
            s($topicname) .
                html_writer::empty_tag('input', [
                    'type' => 'hidden',
                    'name' => 'topic_name[' . $index . ']',
                    'value' => $topicname,
                ]),
            html_writer::checkbox('topic_use_default[' . $index . ']', 1, !$hasoverride, ''),
            html_writer::empty_tag('input', [
                'type' => 'number',
                'name' => 'topic_minimum[' . $index . ']',
                'value' => format_float($topicminimum, 2),
                'min' => '0',
                'max' => '100',
                'step' => '0.01',
                'class' => 'form-control',
                'style' => 'max-width: 140px;',
            ]),
            s(format_float($topicminimum, 1) . '%'),
        ];
    }

    echo html_writer::table($topictable);
}

echo html_writer::empty_tag('input', [
    'type' => 'submit',
    'class' => 'btn btn-primary',
    'value' => get_string('masterypolicysavebutton', 'local_chatbot'),
]);
echo html_writer::end_tag('form');

echo html_writer::tag('h4', get_string('masterypolicysectiondebt', 'local_chatbot'), ['class' => 'mt-4']);
echo html_writer::tag('p', get_string('masterypolicydebtdesc', 'local_chatbot'));
if (!$learningready) {
    echo $OUTPUT->notification(get_string('masterypolicydatamissing', 'local_chatbot'), 'info');
} else if (empty($students)) {
    echo $OUTPUT->notification(get_string('masterypolicynostudents', 'local_chatbot'), 'info');
} else if (empty($topicnames)) {
    echo $OUTPUT->notification(get_string('masterypolicynotopics', 'local_chatbot'), 'info');
} else {
    $debttable = new html_table();
    $debttable->head = [
        get_string('dashboardtablestudent', 'local_chatbot'),
        get_string('masterypolicytopicpassed', 'local_chatbot'),
        get_string('masterypolicytopicdebt', 'local_chatbot'),
        get_string('masterypolicydebtdetail', 'local_chatbot'),
        get_string('masterypolicyexamreadiness', 'local_chatbot'),
        get_string('dashboardtableupdated', 'local_chatbot'),
    ];
    $debttable->data = [];
    foreach ($debtsummaryrows as $row) {
        $profileurl = new moodle_url('/user/profile.php', ['id' => (int)$row->userid]);
        if (empty($policy['requireforexams'])) {
            $statusbadge = html_writer::span(get_string('masterypolicynotenforced', 'local_chatbot'), 'badge badge-secondary');
        } else {
            $statusbadge = $row->ready
                ? html_writer::span(get_string('masterypolicyready', 'local_chatbot'), 'badge badge-success')
                : html_writer::span(get_string('masterypolicyblocked', 'local_chatbot'), 'badge badge-danger');
        }
        $debttable->data[] = [
            html_writer::link($profileurl, s((string)$row->studentname)),
            s((string)$row->passedcount . '/' . (string)$topiccount),
            (int)$row->debtcount,
            s($row->debttopicspreview !== '' ? (string)$row->debttopicspreview : '-'),
            $statusbadge,
            s((int)$row->lastupdate > 0
                ? userdate((int)$row->lastupdate, get_string('strftimedatetime', 'langconfig'))
                : '-'),
        ];
    }

    echo html_writer::table($debttable);
    echo html_writer::tag(
        'p',
        get_string('masterypolicydebtnote', 'local_chatbot'),
        ['class' => 'text-muted mt-2']
    );
}

echo $OUTPUT->footer();
