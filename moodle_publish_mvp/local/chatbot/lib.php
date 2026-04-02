<?php
defined('MOODLE_INTERNAL') || die();

/**
 * Adds LLM Chat page to navigation.
 *
 * @param global_navigation $navigation
 * @return void
 */
function local_chatbot_extend_navigation(global_navigation $navigation): void {
    global $USER;

    if (!isloggedin() || isguestuser()) {
        return;
    }

    $context = context_system::instance();
    if (!has_capability('local/chatbot:view', $context)) {
        return;
    }

    $url = new moodle_url('/local/chatbot/index.php');
    $node = navigation_node::create(
        get_string('pluginname', 'local_chatbot'),
        $url,
        navigation_node::TYPE_CUSTOM,
        null,
        'local_chatbot'
    );
    $navigation->add_node($node);

    require_once(__DIR__ . '/locallib.php');
    $isteacher = local_chatbot_user_is_teacher_like((int)$USER->id) || is_siteadmin((int)$USER->id);
    if ($isteacher) {
        $reporturl = new moodle_url('/local/chatbot/teacher_report.php');
        $reportnode = navigation_node::create(
            get_string('teacherreportlink', 'local_chatbot'),
            $reporturl,
            navigation_node::TYPE_CUSTOM,
            null,
            'local_chatbot_teacher_report'
        );
        $node->add_node($reportnode);
    }
}

/**
 * Add mastery summary nodes to user profile page.
 *
 * @param \core_user\output\myprofile\tree $tree
 * @param stdClass $user
 * @param bool $iscurrentuser
 * @param stdClass|null $course
 * @return void
 */
function local_chatbot_myprofile_navigation(\core_user\output\myprofile\tree $tree, $user, $iscurrentuser, $course): void {
    global $CFG, $USER;

    if (!isloggedin() || isguestuser()) {
        return;
    }

    $systemcontext = context_system::instance();
    if (!has_capability('local/chatbot:view', $systemcontext)) {
        return;
    }

    require_once($CFG->dirroot . '/local/chatbot/locallib.php');
    if (!local_chatbot_learning_tables_ready()) {
        return;
    }

    $topicrows = local_chatbot_get_student_mastery_rows((int)$user->id);
    $classrows = local_chatbot_get_student_class_mastery_rows((int)$user->id);
    $overall = local_chatbot_get_student_overall_mastery((int)$user->id);
    $content = local_chatbot_render_profile_mastery_html(
        $topicrows,
        $classrows,
        $overall,
        $iscurrentuser && (local_chatbot_user_is_teacher_like((int)$USER->id) || is_siteadmin())
    );
    if (trim($content) === '') {
        return;
    }

    $masterynode = new \core_user\output\myprofile\node(
        'reports',
        'chatbotmastery',
        get_string('dashboardtitle', 'local_chatbot'),
        null,
        null,
        $content
    );
    $tree->add_node($masterynode);

    $chaturl = new moodle_url('/local/chatbot/index.php');
    $chatnode = new \core_user\output\myprofile\node(
        'reports',
        'chatbotopen',
        get_string('pluginname', 'local_chatbot'),
        null,
        $chaturl
    );
    $tree->add_node($chatnode);
}

/**
 * Render profile mastery HTML snippet.
 *
 * @param array $topicrows
 * @param array $classrows
 * @param array $overall
 * @param bool $showteacheroverview
 * @return string
 */
function local_chatbot_render_profile_mastery_html(
    array $topicrows,
    array $classrows,
    array $overall,
    bool $showteacheroverview = false
): string {
    global $USER;

    if (empty($topicrows) && empty($classrows) && !$showteacheroverview) {
        return html_writer::tag('p', get_string('dashboardempty', 'local_chatbot'));
    }

    $html = '';
    if (!empty($topicrows) || !empty($classrows)) {
        $html .= html_writer::tag('h4', get_string('dashboardsectionoverall', 'local_chatbot'));
        $overallmetrics = [
            get_string('dashboardcardavgmastery', 'local_chatbot') . ': ' .
                format_float((float)($overall['overallmastery'] ?? 0), 1) . '%',
            get_string('dashboardcardoverallaccuracy', 'local_chatbot') . ': ' .
                format_float((float)($overall['overallaccuracy'] ?? 0), 1) . '%',
            get_string('dashboardcardtrackedclasses', 'local_chatbot') . ': ' . (int)($overall['classcount'] ?? 0),
            get_string('dashboardcardtrackedtopics', 'local_chatbot') . ': ' . (int)($overall['topiccount'] ?? 0),
            get_string('dashboardcardattempts', 'local_chatbot') . ': ' . (int)($overall['attemptsum'] ?? 0),
            get_string('dashboardcardlastupdate', 'local_chatbot') . ': ' .
                ((int)($overall['lastupdate'] ?? 0) > 0
                    ? userdate((int)$overall['lastupdate'], get_string('strftimedatetime', 'langconfig'))
                    : '-'),
        ];
        $html .= html_writer::alist($overallmetrics);

        $html .= html_writer::tag('h4', get_string('dashboardsectionclasses', 'local_chatbot'), ['class' => 'mt-3']);
        $classtable = new html_table();
        $classtable->head = [
            get_string('dashboardtablecourse', 'local_chatbot'),
            get_string('dashboardtablemastery', 'local_chatbot'),
            get_string('dashboardtableaccuracy', 'local_chatbot'),
            get_string('dashboardtabletopics', 'local_chatbot'),
            get_string('dashboardtableattempts', 'local_chatbot'),
            get_string('dashboardtableupdated', 'local_chatbot'),
        ];
        $classtable->data = [];
        foreach ($classrows as $row) {
            $courselabel = trim((string)$row->fullname) !== '' ? (string)$row->fullname : (string)$row->shortname;
            $classtable->data[] = [
                s($courselabel),
                s(format_float((float)$row->classmastery, 1) . '%'),
                s(format_float((float)$row->classaccuracy, 1) . '%'),
                (int)$row->topiccount,
                (int)$row->attemptsum,
                s((int)$row->lastupdate > 0
                    ? userdate((int)$row->lastupdate, get_string('strftimedatetime', 'langconfig'))
                    : '-'),
            ];
        }
        if (empty($classtable->data)) {
            $classtable->data[] = [get_string('dashboardempty', 'local_chatbot'), '', '', '', '', ''];
        }
        $html .= html_writer::table($classtable);

        $html .= html_writer::tag(
            'h4',
            get_string('dashboardsectionstudenttopics', 'local_chatbot'),
            ['class' => 'mt-3']
        );
        $table = new html_table();
        $table->head = [
            get_string('dashboardtablecourse', 'local_chatbot'),
            get_string('dashboardtabletopic', 'local_chatbot'),
            get_string('dashboardtablemastery', 'local_chatbot'),
            get_string('dashboardtableaccuracy', 'local_chatbot'),
            get_string('dashboardtableattempts', 'local_chatbot'),
            get_string('dashboardtableupdated', 'local_chatbot'),
        ];
        $table->data = [];

        foreach (array_slice($topicrows, 0, 20) as $row) {
            $courselabel = trim((string)$row->fullname) !== '' ? (string)$row->fullname : (string)$row->shortname;
            $table->data[] = [
                s($courselabel),
                s((string)$row->topic),
                s(format_float((float)$row->mastery, 1) . '%'),
                s(format_float((float)$row->accuracy_avg, 1) . '%'),
                (int)$row->attempt_count,
                s((int)$row->timemodified > 0
                    ? userdate((int)$row->timemodified, get_string('strftimedatetime', 'langconfig'))
                    : '-'),
            ];
        }
        $html .= html_writer::table($table);
    } else {
        $html .= html_writer::tag('p', get_string('dashboardempty', 'local_chatbot'));
    }

    if ($showteacheroverview) {
        require_once(__DIR__ . '/../../enrol/locallib.php');
        $courseids = [];
        $courses = enrol_get_users_courses((int)$USER->id, true, 'id', 'sortorder ASC');
        foreach ($courses as $course) {
            $coursecontext = context_course::instance((int)$course->id, IGNORE_MISSING);
            if ($coursecontext && has_capability('moodle/course:update', $coursecontext, (int)$USER->id)) {
                $courseids[] = (int)$course->id;
            }
        }

        if (empty($courseids) && is_siteadmin((int)$USER->id)) {
            global $DB;
            $courseids = array_map(
                'intval',
                array_keys($DB->get_records_menu('local_chatbot_std_profile', null, '', 'courseid,courseid'))
            );
        }

        if (!empty($courseids)) {
            $overview = local_chatbot_get_teacher_mastery_dashboard($courseids);
            if ((int)$overview['summary']['profilecount'] > 0) {
                $teacheritems = [
                    get_string('dashboardcardstudents', 'local_chatbot') . ': ' . (int)$overview['summary']['studentcount'],
                    get_string('dashboardcardprofiles', 'local_chatbot') . ': ' . (int)$overview['summary']['profilecount'],
                    get_string('dashboardcardavgmastery', 'local_chatbot') . ': ' .
                        format_float((float)$overview['summary']['avgmastery'], 1) . '%',
                    get_string('dashboardcardevents', 'local_chatbot') . ': ' . (int)$overview['summary']['eventcount'],
                ];
                $html .= html_writer::tag(
                    'h4',
                    get_string('dashboardsubtitleteacher', 'local_chatbot'),
                    ['class' => 'mt-3']
                );
                $html .= html_writer::alist($teacheritems);

                $html .= html_writer::tag(
                    'h4',
                    get_string('dashboardsectionteacherstudents', 'local_chatbot'),
                    ['class' => 'mt-3']
                );

                $studenttable = new html_table();
                $studenttable->head = [
                    get_string('dashboardtablecourse', 'local_chatbot'),
                    get_string('dashboardtablestudent', 'local_chatbot'),
                    get_string('dashboardtablemastery', 'local_chatbot'),
                    get_string('dashboardtableaccuracy', 'local_chatbot'),
                    get_string('dashboardtableattempts', 'local_chatbot'),
                    get_string('dashboardtableupdated', 'local_chatbot'),
                ];
                $studenttable->data = [];

                foreach (array_slice((array)$overview['learners'], 0, 30) as $row) {
                    $courselabel = trim((string)$row->fullname) !== '' ? (string)$row->fullname : (string)$row->shortname;
                    $studentname = trim((string)$row->firstname . ' ' . (string)$row->lastname);
                    if ($studentname === '') {
                        $studentname = get_string('unknownuser', 'moodle');
                    }
                    $profileurl = new moodle_url('/user/profile.php', ['id' => (int)$row->userid]);

                    $studenttable->data[] = [
                        s($courselabel),
                        html_writer::link($profileurl, s($studentname)),
                        s(format_float((float)$row->avgmastery, 1) . '%'),
                        s(format_float((float)$row->avgaccuracy, 1) . '%'),
                        (int)$row->attemptsum,
                        s((int)$row->lastupdate > 0
                            ? userdate((int)$row->lastupdate, get_string('strftimedatetime', 'langconfig'))
                            : '-'),
                    ];
                }

                if (empty($studenttable->data)) {
                    $studenttable->data[] = [get_string('dashboardempty', 'local_chatbot'), '', '', '', '', ''];
                }

                $html .= html_writer::table($studenttable);

                $reporturl = new moodle_url('/local/chatbot/teacher_report.php');
                $html .= html_writer::tag(
                    'p',
                    html_writer::link($reporturl, get_string('teacherreportlink', 'local_chatbot'))
                );
            }
        }
    }

    return $html;
}
