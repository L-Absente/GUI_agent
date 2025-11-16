


SETTING_TASKS_1 = """**Task Categories with Explanations and Examples:**

1. **User Authentication** - Screens focused on verifying user identity and managing access permissions
   Examples: Login pages, registration forms, password reset interfaces, two-factor authentication setups, account recovery flows, single sign-on (SSO) screens

2. **Data Management** - Interfaces designed for creating, reading, modifying, or deleting data records and content
   Examples: CRUD forms, data tables with edit/delete actions, file upload/download interfaces, database administration panels, content management systems, data import/export wizards

3. **Analytics & Reporting** - Screens dedicated to visualizing data, generating insights, and presenting analytical information
   Examples: Dashboard charts and graphs, report generation interfaces, data filtering and aggregation tools, KPI monitoring displays, trend analysis visualizations, business intelligence dashboards

4. **System Configuration** - Interfaces for setting up, customizing, and managing system parameters, preferences, and integrations
   Examples: Settings panels, user role and permission management, system integration configurations, notification setup screens, API key management interfaces, application preference dialogs

5. **Navigation & Workflow** - Screens that guide users through multi-step processes, provide navigation structures, or facilitate task completion flows
   Examples: Onboarding tutorials, checkout processes, wizard interfaces, menu systems, breadcrumb navigation, task management interfaces, workflow automation setups"""

SETTING_TASKS_2 = """**Task Categories with Clear Definitions and Specific Examples:**

1. **Authentication & Security** - Screens exclusively focused on user identity verification and access control
   *Must contain: login fields, registration forms, or security verification elements*
   Examples: Username/password login screens, biometric authentication prompts, password change forms, 2FA setup pages, account lockout screens

2. **Data Creation & Editing** - Interfaces primarily for inputting, modifying, or deleting specific data records
   *Must contain: form fields, input controls, or edit/delete actions for specific data items*
   Examples: User profile edit forms, product creation dialogs, document editors, data entry forms, record update interfaces

3. **Data Visualization & Analysis** - Screens dominated by charts, graphs, tables, or analytical tools
   *Must contain: visual data representations like charts, graphs, or analytical dashboards*
   Examples: Sales dashboard with charts, analytics reports with graphs, data monitoring displays, business intelligence dashboards, statistical visualization interfaces

4. **System Configuration** - Interfaces for modifying application settings, preferences, or system parameters
   *Must contain: settings toggles, configuration options, or system management controls*
   Examples: Application settings panels, user permission management screens, notification configuration pages, API settings interfaces, system preference dialogs

5. **Content Browsing & Navigation** - Screens primarily for viewing content, navigating between sections, or browsing information
   *Must contain: content listings, navigation menus, or browsing interfaces without heavy data manipulation*
   Examples: Main application menus, content category listings, file browsers, document libraries, news/article feeds, search result pages"""

def generate_prompt(instruction):
    """
    Generate a prompt that combines the given instruction with GUI screenshot analysis
    for task category classification.
    
    Args:
        instruction (str): The instruction from the dataset that describes the task
        
    Returns:
        str: The generated prompt with both instruction and GUI analysis requirements
    """
    return f"""You are an expert GUI classifier. Your task is to analyze the provided screenshot of a graphical user interface along with the user instruction, and classify the interaction into one of the predefined task categories based on both the visual interface and the intended action.

**User Instruction:**
"{instruction}"

{SETTING_TASKS_2}

**Classification Instructions:**
1. Carefully examine both the GUI screenshot AND the user instruction together
2. Consider how the instruction relates to the visible interface elements and their functionality
3. Identify which main category (1-5 above) best describes the combined context of the instruction and GUI
4. Consider the examples provided for each category to understand the scope and boundaries
5. Provide detailed reasoning that references:
   - Specific visual elements in the GUI
   - How the instruction interacts with or relates to these elements
   - Why this category is the best fit for the combined instruction-GUI context
6. Follow the exact output format specified below

**Output Format:**
- If the GUI and instruction clearly belong to one of the predefined categories:
Classification: [Category Name]
Reasoning: [Detailed explanation referencing specific UI elements, the instruction content, and how they work together to justify this classification]

- If the combination does not fit well into any predefined category:
Classification: Other
Suggested Category: [Proposed category name that captures the primary function]
Reasoning: [Explanation of why existing categories don't fit and how your suggested category better describes this instruction-GUI combination]

**Critical Guidelines:**
- Focus on the PRIMARY function that emerges from combining the instruction with the GUI interface
- Use the examples as concrete guides for category boundaries - if the instruction-GUI combination looks and functions like the examples, it belongs in that category
- Base your reasoning on visible elements only: buttons, text fields, navigation menus, charts, forms, labels, and overall layout, combined with the instruction's intent
- Do not make assumptions about backend functionality that isn't visually apparent or explicitly stated in the instruction
- If multiple categories seem plausible, choose the one whose examples most closely match the combined purpose of the instruction and GUI
- Be specific in your reasoning - reference actual UI elements you can see AND how the instruction relates to them
- Maintain objectivity and avoid personal interpretations not supported by visual evidence or the instruction content
- The instruction provides context for how the user intends to interact with the interface, which may change the classification from what the GUI alone would suggest

**Important:** Your output must follow the exact format specified above. Only output the classification result and reasoning - no additional text, explanations, or formatting.

Analyze both the instruction and the screenshot thoroughly and provide your classification now."""
