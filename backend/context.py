def build_meal_planning_prompt(user_context: dict) -> str:
    """Build a meal planning prompt based on user context."""
    if not user_context:
        return (
            "User's intent is Meal-Planning-Recipes. "
            "No user context provided. Suggest a general meal."
        )
    
    allergies = ', '.join(user_context.get("allergies", [])) or "None"
    likes = ', '.join(user_context.get("likes", [])) or "No specific preferences"
    dislikes = ', '.join(user_context.get("dislikes", [])) or "None"
    diet = user_context.get("diet", "No Restriction")
    goal = user_context.get("goal", "Maintain Weight")
    
    prompt = (
        f"You are a nutritionist creating meal plans. Critical constraints:\n"
        f"- ABSOLUTELY NEVER suggest {allergies} (life-threatening allergies)\n"
        f"- Avoid: {dislikes}\n"
        f"- Prioritize: {likes}\n"
        f"- Diet: {diet}\n"
        f"- Goal: {goal}\n\n"
        "Thinking process:\n"
        "1. Check allergies first - remove prohibited ingredients\n"
        "2. Eliminate disliked foods\n"
        "3. Select preferred ingredients\n"
        f"4. Calculate nutritional needs based on {goal}\n"
        "5. Propose a meal meeting all criteria"
    )
    
    return prompt