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

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'LLM Chat';
$string['settingsheading'] = 'LLM Chat settings';
$string['projectpath'] = 'Project path';
$string['projectpath_desc'] = 'Absolute path to the LLM project folder.';
$string['pythonpath'] = 'Python executable path';
$string['pythonpath_desc'] = 'Absolute path to python executable used by the plugin.';
$string['runnerfile'] = 'Runner file name';
$string['runnerfile_desc'] = 'Backend runner filename inside the project path.';

$string['chatbot:view'] = 'View LLM Chat page.';
$string['chatbot:managedrafts'] = 'Create and update generated drafts.';
$string['chatbot:publish'] = 'Publish generated draft to course.';
$string['chatbot:manageweights'] = 'Manage weighting scheme UI and activity mapping.';

$string['uploadtitle'] = 'Insert PDF/TXT';
$string['uploaddesc'] = 'Select class and topic to load materials from course resources.';
$string['uploadbutton'] = 'Upload selected files';
$string['uploadedtitle'] = 'Materials';
$string['nofiles'] = 'No materials found for selected class/topic.';
$string['statusready'] = 'RAG ready';
$string['statusnodocs'] = 'No materials selected';
$string['chatheader'] = 'Chat with class materials';
$string['chatplaceholder'] = 'Ask a question about selected class materials...';
$string['chatusagelabel'] = 'Usage';
$string['clearhistorylabel'] = 'Clear history';
$string['clearhistoryconfirm'] = 'Clear this chat history?';
$string['defaultgreeting'] = 'Hello. Select class and topic in the left panel, then ask about the materials.';
$string['thinking'] = 'Thinking...';
$string['uploading'] = 'Uploading...';
$string['uploadfailed'] = 'Upload failed.';
$string['chaterror'] = 'Failed to process chat request.';
$string['previewtitle'] = 'Preview Draft';
$string['previewempty'] = 'No preview available.';
$string['previewloading'] = 'Generating preview...';
$string['previewerror'] = 'Failed to generate preview.';

$string['tabchat'] = 'Chat';
$string['tabassignment'] = 'Assignment';
$string['tabpractice'] = 'Practice';
$string['tabdashboard'] = 'Mastery';
$string['roleteacheronly'] = 'This feature is available for teachers only.';

$string['dashboardtitle'] = 'Learning Mastery Dashboard';
$string['dashboardsubtitlestudent'] = 'Your mastery progression by class and topic.';
$string['dashboardsubtitleteacher'] = 'Monitor mastery trend across classes and learners.';
$string['dashboardempty'] = 'No mastery data yet. Submit quiz/practice attempts first.';
$string['dashboardcardavgmastery'] = 'Average mastery';
$string['dashboardcardoverallaccuracy'] = 'Overall accuracy';
$string['dashboardcardtrackedclasses'] = 'Tracked classes';
$string['dashboardcardtrackedtopics'] = 'Tracked topics';
$string['dashboardcardattempts'] = 'Total attempts';
$string['dashboardcardlastupdate'] = 'Last update';
$string['dashboardcardstudents'] = 'Students tracked';
$string['dashboardcardprofiles'] = 'Topic profiles';
$string['dashboardcardevents'] = 'Learning events';
$string['dashboardtablecourse'] = 'Course';
$string['dashboardtabletopic'] = 'Topic';
$string['dashboardtablemastery'] = 'Mastery';
$string['dashboardtableaccuracy'] = 'Accuracy';
$string['dashboardtableattempts'] = 'Attempts';
$string['dashboardtabletopics'] = 'Topics';
$string['dashboardtableduration'] = 'Avg duration';
$string['dashboardtabletrend'] = 'Trend';
$string['dashboardtablelastscore'] = 'Last score';
$string['dashboardtableupdated'] = 'Updated';
$string['dashboardtablelearners'] = 'Learners';
$string['dashboardtablestudent'] = 'Student';
$string['dashboardfilterstudent'] = 'Student filter';
$string['dashboardtabletype'] = 'Type';
$string['dashboardtablescore'] = 'Score';
$string['dashboardtabletime'] = 'Time';
$string['dashboardtypepractice'] = 'Practice';
$string['dashboardtypequiz'] = 'Quiz';
$string['dashboardsectionstudenttopics'] = 'Mastery and progress by topic';
$string['dashboardsectionoverall'] = 'Your overall mastery';
$string['dashboardsectionclasses'] = 'Your mastery by class';
$string['dashboardsectiontopicrisk'] = 'Topic risk overview';
$string['dashboardsectionlearnerrisk'] = 'Learner risk overview';
$string['dashboardsectionevents'] = 'Recent learning events';
$string['dashboardsectionteacherstudents'] = 'Student progress details';
$string['dashboardsectionprogressinsights'] = 'Progress insights by topic';
$string['dashboardsectionteachertopicprogress'] = 'Student-topic progress insights';
$string['dashboardtablemasterydelta'] = 'Mastery change';
$string['dashboardtablefirstattempt'] = 'First-attempt accuracy';
$string['dashboardtabletimetotarget'] = 'Time to target (75%)';
$string['dashboardtabletrendchart'] = 'Daily trend';
$string['dashboardtargetnotreached'] = 'Not reached';

$string['assignmenttitle'] = 'Create assignment with LLM';
$string['assignmentdesc'] = 'Generate assignment draft, review it, then publish to target class.';
$string['assignmentclass'] = 'Target class';
$string['assignmentclassplaceholder'] = 'Select class';
$string['assignmentnocourses'] = 'No teacher course found';
$string['nocoursesavailable'] = 'No enrolled course found';
$string['assignmenttopic'] = 'Topic';
$string['assignmenttopicplaceholder'] = 'Select topic';
$string['assignmenttopicloading'] = 'Loading topics...';
$string['assignmenttopicempty'] = 'No topics found in this class';
$string['assignmentpdf'] = 'Material';
$string['assignmentpdfplaceholder'] = 'Select PDF';
$string['assignmentpdfloading'] = 'Loading PDFs...';
$string['assignmentpdfempty'] = 'No PDF resource found in this class';
$string['assignmenttype'] = 'Assignment type';
$string['assignmenttypeindividual'] = 'Individual assignment';
$string['assignmenttypegroup'] = 'Group assignment';
$string['assignmenttypesummary'] = 'Summary assignment';
$string['assignmenttypepresentation'] = 'Presentation assignment';
$string['assignmenttypelabreport'] = 'Lab report assignment';
$string['assignmenttypecustom'] = 'Custom assignment type';
$string['assignmenttypequiz'] = 'Quiz (multiple choice)';
$string['assignmenttypecustomlabel'] = 'Custom type label';
$string['assignmenttypecustomplaceholder'] = 'Example: Poster reflection assignment';
$string['assignmentweightlabel'] = 'Weight label';
$string['assignmentweightlabel_help'] = 'Choose the difficulty-weight label for this activity. The system converts the label into percent automatically.';
$string['assignmentweightveryeasy'] = 'Very Easy';
$string['assignmentweighteasy'] = 'Easy';
$string['assignmentweightmedium'] = 'Medium';
$string['assignmentweighthard'] = 'Hard';
$string['assignmentweightveryhard'] = 'Very Hard';
$string['assignmentweightpercent'] = 'Weight (%)';
$string['assignmentcount'] = 'Number of questions/components';
$string['assignmentnotes'] = 'Additional notes';
$string['assignmentnotesplaceholder'] = 'Add constraints, output format, deadline policy, or custom rubric.';
$string['assignmentgenerate'] = 'Generate';
$string['assignmentregenerate'] = 'Regenerate';
$string['assignmentpublish'] = 'Publish';
$string['assignmentpublished'] = 'Published';
$string['assignmentpublishing'] = 'Publishing...';
$string['assignmentpublisherror'] = 'Failed to publish draft.';
$string['assignmentgeneratedfirst'] = 'Generate draft first before publishing.';
$string['assignmentselectclassfirst'] = 'Select target class first.';
$string['assignmentpreview'] = 'Assignment draft preview';
$string['assignmentplaceholder'] = 'Draft will appear here after generation.';
$string['assignmentessayautograde'] = 'Enable auto-grading for essay submissions';
$string['teacherreporttitle'] = 'Teacher Mastery Report';
$string['teacherreportsubtitle'] = 'Simple mastery view for each student.';
$string['teacherreportlink'] = 'Mastery report';
$string['teacherreportfiltercourse'] = 'Class filter';
$string['teacherreportallcourses'] = 'All my classes';
$string['teacherreportempty'] = 'No mastery data for selected class filter.';
$string['teacherreportclasses'] = 'Classes';
$string['teacherreporttopicprogress'] = 'Topic progress report';

$string['practicetitle'] = 'Practice generator';
$string['practicedesc'] = 'Generate practice questions from selected topic.';
$string['practicegenerate'] = 'Generate practice';
$string['practicepublish'] = 'Publish practice';
$string['practicepublished'] = 'Practice published';
$string['practicepublishing'] = 'Publishing practice...';
$string['practicepublisherror'] = 'Failed to publish practice draft.';
$string['practicegeneratedfirst'] = 'Generate practice draft first before publishing.';
$string['practicepublishsavedstudent'] = 'Practice saved to your profile. Ask your teacher to publish it as a Moodle quiz if needed.';
$string['practicepreview'] = 'Practice preview';
$string['practiceplaceholder'] = 'Practice output will appear here.';

$string['savedraftsuccess'] = 'Draft saved successfully.';
$string['publishsuccess'] = 'Draft published successfully to class.';
$string['practicepublishsuccess'] = 'Practice quiz published successfully to class.';
$string['draftcoursemismatch'] = 'Draft does not belong to the selected course.';
$string['publishfailed'] = 'Publish failed while creating Moodle activity.';
$string['missingdraftpayload'] = 'No draft payload was provided.';
$string['invaliddraftjson'] = 'Draft JSON is invalid.';
$string['missingdraftsection'] = 'Draft is missing required section: {$a}.';
$string['invalidassignmenttitle'] = 'Assignment title is empty or invalid.';
$string['invalidquestions'] = 'Question list is empty or invalid.';
$string['questioncountmismatch'] = 'Question count does not match configured value.';
$string['invalidquestionstem'] = 'Question stem is invalid at number {$a}.';
$string['invalidquestionoptions'] = 'Question options A-D are invalid at number {$a}.';
$string['invalidanswerkey'] = 'Answer key is invalid.';
$string['invalidanswerkeynumber'] = 'Answer key is missing question number {$a}.';
$string['invalidanswerkeyletter'] = 'Answer key letter is invalid at question {$a}.';

$string['weightsettingslink'] = 'Weight settings';
$string['masterypolicylink'] = 'Mastery policy';
$string['weightsettingstitle'] = 'Assessment Weight Settings';
$string['weightsettingssubtitle'] = 'Configure category/type/source weighting and map each activity bucket.';
$string['weightsautodefaultnotice'] = 'New Assignment/Quiz created from course page is auto-mapped with default weights. Review and adjust below when needed.';
$string['weightsettingsnocourse'] = 'No manageable course found for weight settings.';
$string['weightstablemissing'] = 'Weight settings tables are not available yet. Run Moodle upgrade first.';
$string['nativeweightingheading'] = 'LLM mastery mapping';
$string['nativeweightingactivitytype'] = 'Assessment type';
$string['nativeweightingactivitytype_help'] = 'Select how this activity should be classified for mastery weighting (for example: Individual, Group, Quiz, UTS, UAS).';

$string['weightsstepsheading'] = 'Recommended setup flow';
$string['weightsstep1'] = 'Set category weights (Task and Exam) with total 100%.';
$string['weightsstep2'] = 'Set type weights inside each category with total 100%.';
$string['weightsstep3'] = 'Set source weights (Teacher and LLM) for each type with total 100%.';
$string['weightsstep4'] = 'Map each assignment/quiz activity to category, type, and source.';

$string['weightsrulesheading'] = 'Step 1: Weight rules';
$string['weightcategoryheading'] = 'Category weight';
$string['weighttypeheading'] = 'Type weight';
$string['weightsourceheading'] = 'Source weight';
$string['weightmapheading'] = 'Step 2: Activity mapping';
$string['weightpreviewheading'] = 'Step 3: Effective weight preview';
$string['weightpreviewformula'] = 'Effective base weight = category% * type% * source%.';

$string['weightcategorytask'] = 'Task';
$string['weightcategoryexam'] = 'Exam';
$string['weighttypeindividual'] = 'Individual Assignment';
$string['weighttypegroup'] = 'Group Assignment';
$string['weighttypepractice'] = 'Practice';
$string['weighttypequiz'] = 'Quiz';
$string['weighttypeuts'] = 'UTS';
$string['weighttypeuas'] = 'UAS';
$string['weightsourceteacher'] = 'Teacher';
$string['weightsourcellm'] = 'LLM';

$string['weighttablecategory'] = 'Category';
$string['weighttabletype'] = 'Type';
$string['weighttablesource'] = 'Source';
$string['weighttableweightpercent'] = 'Weight (%)';
$string['weighttableactivity'] = 'Activity';
$string['weighttablemodule'] = 'Module';
$string['weighttableitemweightlabel'] = 'Item weight label';
$string['weighttableitemweight'] = 'Item weight (%)';
$string['weighttableeffective'] = 'Effective base weight';
$string['weighttablemappedactivities'] = 'Mapped activities';
$string['weightmaplabelnote'] = 'Select a weight label per activity; the percent is auto-derived from the label.';

$string['weightsavescheme'] = 'Save weight rules';
$string['weightsavemap'] = 'Save activity mapping';
$string['weightssaved'] = 'Weight rules saved.';
$string['weightmapsaved'] = 'Activity mapping saved ({$a} row(s)).';
$string['weightmapemptyactivities'] = 'No assign/quiz activity found in this course yet.';

$string['weightserrorcategorysum'] = 'Category total must be 100%. Current total: {$a}.';
$string['weightserrortypesum'] = '{$a->bucket} type total must be 100%. Current total: {$a->sum}.';
$string['weightserrorsourcesum'] = '{$a->bucket} source total must be 100%. Current total: {$a->sum}.';
$string['weightserrormapinvalidcategory'] = 'Invalid category selected on activity CMID {$a}.';
$string['weightserrormapinvalidtype'] = 'Invalid type selected on activity CMID {$a}.';
$string['weightserrormapinvalidsource'] = 'Invalid source selected on activity CMID {$a}.';

$string['masterypolicytitle'] = 'Mastery Policy and Topic Completion';
$string['masterypolicysubtitle'] = 'Set minimum mastery per topic and monitor learner not-passed status for exam readiness.';
$string['masterypolicynocourse'] = 'No manageable course found for mastery policy.';
$string['masterypolicysectionrules'] = 'Step 1: Topic minimum mastery policy';
$string['masterypolicydefaultminimum'] = 'Default minimum mastery (%)';
$string['masterypolicyrequireexam'] = 'Require all previous topics to pass minimum mastery before UTS/UAS.';
$string['masterypolicyusedefault'] = 'Use default minimum';
$string['masterypolicyoverride'] = 'Override minimum (%)';
$string['masterypolicyeffective'] = 'Effective minimum';
$string['masterypolicysavebutton'] = 'Save mastery policy';
$string['masterypolicysaved'] = 'Mastery policy saved.';
$string['masterypolicynotopics'] = 'No topic section found in this class yet.';
$string['masterypolicysectiondebt'] = 'Step 2: Topic not-passed monitor';
$string['masterypolicydebtdesc'] = 'Student can continue to the next topic, but unresolved not-passed topics will block UTS/UAS when exam gate is active.';
$string['masterypolicydatamissing'] = 'Learning profile tables are not available yet. Run Moodle upgrade first.';
$string['masterypolicynostudents'] = 'No enrolled learners found in this class.';
$string['masterypolicytopicpassed'] = 'Topics passed';
$string['masterypolicytopicdebt'] = 'Topic not passed';
$string['masterypolicydebtdetail'] = 'Not-passed topics';
$string['masterypolicyexamreadiness'] = 'UTS/UAS readiness';
$string['masterypolicyready'] = 'Ready';
$string['masterypolicyblocked'] = 'Blocked';
$string['masterypolicynotenforced'] = 'Not enforced';
$string['masterypolicydebtnote'] = 'Exam gate applies to activities mapped as UTS/UAS when the policy toggle is enabled and there are topics not passed.';
$string['masterygateblockedmessage'] = 'UTS/UAS attempt is blocked. You still have {$a->count} topic(s) not passed: {$a->topics}. Complete pending assignments/quizzes, remedial, or practice first.';

$string['coursemasterywidgettitle'] = 'Your mastery by topic';
$string['coursemasterysummary'] = 'Topik lulus: {$a->passed}/{$a->total}. Topik tidak lulus: {$a->notpassed}. Topik berjalan: {$a->inprogress}.';
$string['coursemasterytabletopic'] = 'Topic';
$string['coursemasterytablemastery'] = 'Your mastery';
$string['coursemasterytableminimum'] = 'Minimum mastery';
$string['coursemasterytablestatus'] = 'Status';
$string['coursemasterystatuspassed'] = 'LULUS';
$string['coursemasterystatusdebt'] = 'TIDAK LULUS';
$string['coursemasterystatuscurrent'] = 'TOPIK BERJALAN';
$string['coursemasterynodata'] = 'No data yet';
