


SETTING_TASKS = """**Task Categories with Explanations and Examples:**

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


prompt_1 = f"""You are an expert GUI classifier. Your task is to analyze the provided screenshot of a graphical user interface and classify it into one of the predefined task categories based on its primary purpose and functionality.

{SETTING_TASKS}

**Classification Instructions:**
1. Carefully examine the GUI screenshot, focusing on the primary purpose, layout, and key interactive elements
2. Identify which main category (1-5 above) best describes the screenshot's primary function
3. Consider the examples provided for each category to understand the scope and boundaries
4. Provide detailed reasoning that references specific visual elements and explains why this category is the best fit
5. Follow the exact output format specified below

**Output Format:**
- If the GUI clearly belongs to one of the predefined categories:
Classification: [Category Name]
Reasoning: [Detailed explanation referencing specific UI elements, layout patterns, and functional purpose that justify this classification]

- If the GUI does not fit well into any predefined category:
Classification: Other
Suggested Category: [Proposed category name that captures the primary function]
Reasoning: [Explanation of why existing categories don't fit and how your suggested category better describes this interface]

**Critical Guidelines:**
- Focus on the PRIMARY function of the interface, not secondary features
- Use the examples as concrete guides for category boundaries - if it looks and functions like the examples, it belongs in that category
- Base your reasoning on visible elements only: buttons, text fields, navigation menus, charts, forms, labels, and overall layout
- Do not make assumptions about backend functionality that isn't visually apparent
- If multiple categories seem plausible, choose the one whose examples most closely match the screenshot's purpose
- Be specific in your reasoning - reference actual UI elements you can see (e.g., "login button and password field" not just "authentication elements")
- Maintain objectivity and avoid personal interpretations not supported by visual evidence

**Important:** Your output must follow the exact format specified above. Only output the classification result and reasoning - no additional text, explanations, or formatting.

Analyze the screenshot thoroughly and provide your classification now."""
